# agents/dqn_agent.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.shared.cnn_actor_critic import ActorCriticCNN


# ----------------------------
# Replay Buffer (contiguous uint8 storage)
# ----------------------------
@dataclass
class _ReplayBatch:
    s: torch.Tensor
    a: torch.Tensor
    r: torch.Tensor
    d: torch.Tensor
    s2: torch.Tensor
    mask: torch.Tensor        # (B, A) bool for current state (optional usage)
    next_mask: torch.Tensor   # (B, A) bool used for masked target


class ReplayBuffer:
    """
    고속/저메모리 Replay:
    - state/next_state: uint8 (0..255), shape = (cap, C, H, W)
    - masks: bool (cap, A)
    - 파이썬 list + np.stack 제거 (샘플링 CPU 오버헤드 크게 감소)
    """

    def __init__(self, capacity: int, num_actions: int):
        self.cap = int(max(1, capacity))
        self.num_actions = int(max(1, num_actions))

        self.ptr = 0
        self.size = 0

        # lazy alloc (first add()에서 state shape 확인 후 할당)
        self._inited = False
        self._C = 0
        self._H = 0
        self._W = 0

        self.states_u8: Optional[np.ndarray] = None       # (cap, C, H, W) uint8
        self.next_states_u8: Optional[np.ndarray] = None  # (cap, C, H, W) uint8

        self.actions = np.zeros((self.cap,), dtype=np.int64)
        self.rewards = np.zeros((self.cap,), dtype=np.float32)
        self.dones = np.zeros((self.cap,), dtype=np.float32)  # 1.0 if done else 0.0

        # masks: None을 허용하지 않고, None이면 all True로 저장
        self.masks = np.ones((self.cap, self.num_actions), dtype=np.bool_)
        self.next_masks = np.ones((self.cap, self.num_actions), dtype=np.bool_)

    @staticmethod
    def _to_u8(obs_f32: np.ndarray) -> np.ndarray:
        x = np.asarray(obs_f32, dtype=np.float32)
        x = np.clip(x, 0.0, 1.0)
        return (x * 255.0 + 0.5).astype(np.uint8, copy=False)

    def _ensure_init(self, s_u8: np.ndarray) -> None:
        if self._inited:
            return
        su = np.asarray(s_u8, dtype=np.uint8)
        if su.ndim != 3:
            raise ValueError(f"ReplayBuffer expects (C,H,W), got {su.shape}")
        C, H, W = int(su.shape[0]), int(su.shape[1]), int(su.shape[2])
        self._C, self._H, self._W = C, H, W
        self.states_u8 = np.zeros((self.cap, C, H, W), dtype=np.uint8)
        self.next_states_u8 = np.zeros((self.cap, C, H, W), dtype=np.uint8)
        self._inited = True

    def __len__(self) -> int:
        return int(self.size)

    def add(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        d: bool,
        s2: np.ndarray,
        action_mask: Optional[np.ndarray],
        next_action_mask: Optional[np.ndarray],
    ) -> None:
        s_u8 = self._to_u8(s)
        s2_u8 = self._to_u8(s2)
        self._ensure_init(s_u8)

        i = int(self.ptr)

        # store
        assert self.states_u8 is not None and self.next_states_u8 is not None
        self.states_u8[i] = s_u8
        self.next_states_u8[i] = s2_u8

        self.actions[i] = int(a)
        self.rewards[i] = float(r)
        self.dones[i] = 1.0 if bool(d) else 0.0

        # masks normalize -> if None or invalid, all True
        self.masks[i] = self._normalize_mask(action_mask)
        self.next_masks[i] = self._normalize_mask(next_action_mask)

        self.ptr = (i + 1) % self.cap
        self.size = min(self.cap, self.size + 1)

    def _normalize_mask(self, mask: Optional[np.ndarray]) -> np.ndarray:
        if mask is None:
            return np.ones((self.num_actions,), dtype=np.bool_)
        m = np.asarray(mask, dtype=np.bool_)
        if m.shape != (self.num_actions,):
            return np.ones((self.num_actions,), dtype=np.bool_)
        if not bool(m.any()):
            return np.ones((self.num_actions,), dtype=np.bool_)
        return m

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = int(self.size)
        if n <= 0:
            raise RuntimeError("ReplayBuffer.sample() called with empty buffer")

        bs = int(min(max(1, batch_size), n))
        idxs = np.random.randint(0, n, size=(bs,), dtype=np.int64)

        assert self.states_u8 is not None and self.next_states_u8 is not None
        s = self.states_u8[idxs]        # (B,C,H,W) uint8
        s2 = self.next_states_u8[idxs]  # (B,C,H,W) uint8
        a = self.actions[idxs]
        r = self.rewards[idxs]
        d = self.dones[idxs]
        m = self.masks[idxs]            # (B,A) bool
        nm = self.next_masks[idxs]      # (B,A) bool
        return s, a, r, d, s2, m, nm

    def clear(self) -> None:
        self.ptr = 0
        self.size = 0
        # 데이터는 남겨도 되지만, 안전하게 masks는 True로 초기화
        self.masks[:] = True
        self.next_masks[:] = True


# ----------------------------
# DQN Agent
# ----------------------------
class DQNAgent:
    """
    runner 호환:
    - select_action(state, action_mask) -> (action_idx, 0.0, 0.0)
    - store(..., next_state=..., next_action_mask=...)
    - should_update() / update()
    - save/load
    """

    def __init__(
        self,
        input_channels: int,
        num_actions: int,
        obs_channels_per_frame: int = 4,

        lr: float = 2e-4,
        gamma: float = 0.99,
        batch_size: int = 128,

        replay_size: int = 20_000,
        learning_starts: int = 2000,

        train_every_steps: int = 16,
        target_update_every_steps: int = 2000,

        double_dqn: bool = True,

        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 200_000,

        max_grad_norm: float = 10.0,

        device: Optional[str] = None,
        use_amp: bool = True,
        cudnn_benchmark: bool = True,
        channels_last: bool = False,
        pin_memory: bool = True,

        # ✅ NEW: gradient accumulation
        grad_accum_steps: int = 1,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(use_amp and (self.device == "cuda"))
        self.pin_memory = bool(pin_memory and (self.device == "cuda"))
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

        self.num_actions = int(num_actions)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)

        self.learning_starts = int(learning_starts)
        self.train_every_steps = int(max(1, train_every_steps))
        self.target_update_every_steps = int(max(1, target_update_every_steps))
        self.double_dqn = bool(double_dqn)

        self.max_grad_norm = float(max_grad_norm)

        # ✅ grad accumulation
        self.grad_accum_steps = int(max(1, grad_accum_steps))

        # epsilon schedule
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay_steps = int(max(1, eps_decay_steps))

        self.global_step = 0   # env steps (store 호출마다 +1)
        self.update_step = 0   # optimizer step count

        # Q network: ActorCriticCNN의 logits를 Q값으로 사용
        self.q = ActorCriticCNN(
            input_channels=int(input_channels),
            num_actions=int(num_actions),
            obs_channels_per_frame=int(obs_channels_per_frame),
            meta_patch=4,
            meta_channel_offset=0,
        ).to(self.device)

        self.q_tgt = ActorCriticCNN(
            input_channels=int(input_channels),
            num_actions=int(num_actions),
            obs_channels_per_frame=int(obs_channels_per_frame),
            meta_patch=4,
            meta_channel_offset=0,
        ).to(self.device)

        self.channels_last = bool(channels_last and self.device == "cuda")
        if self.channels_last:
            self.q = self.q.to(memory_format=torch.channels_last)
            self.q_tgt = self.q_tgt.to(memory_format=torch.channels_last)

        self.q_tgt.load_state_dict(self.q.state_dict(), strict=False)
        self.q_tgt.eval()

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=float(lr))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.replay = ReplayBuffer(capacity=int(replay_size), num_actions=self.num_actions)

    # ---------- epsilon ----------
    def _epsilon(self) -> float:
        t = min(self.global_step, self.eps_decay_steps)
        frac = 1.0 - (t / float(self.eps_decay_steps))
        return float(self.eps_end + (self.eps_start - self.eps_end) * frac)

    # ---------- action mask helpers ----------
    @staticmethod
    def _normalize_mask(mask: Optional[np.ndarray], num_actions: int) -> np.ndarray:
        if mask is None:
            return np.ones((num_actions,), dtype=np.bool_)
        m = np.asarray(mask, dtype=np.bool_)
        if m.shape == (num_actions,) and bool(m.any()):
            return m
        return np.ones((num_actions,), dtype=np.bool_)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, action_mask: np.ndarray | None = None):
        """
        epsilon-greedy + action mask.
        반환 형식: (action, 0.0, 0.0)  (runner 호환)
        """
        m = self._normalize_mask(action_mask, self.num_actions)
        allowed = np.flatnonzero(m)
        if allowed.size <= 0:
            allowed = np.arange(self.num_actions, dtype=np.int64)

        eps = self._epsilon()
        if random.random() < eps:
            a = int(np.random.choice(allowed))
            return a, 0.0, 0.0

        s = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)  # (1,C,H,W)
        if self.pin_memory and s.device.type == "cpu" and self.device == "cuda":
            s = s.pin_memory()
        s = s.to(self.device, non_blocking=self.pin_memory, dtype=torch.float32)

        if s.is_cuda and s.dim() == 4 and self.channels_last:
            s = s.contiguous(memory_format=torch.channels_last)

        q_logits, _v = self.q(s)        # (1,A)
        qv = q_logits.squeeze(0)        # (A,)

        # ✅ AMP/FP16 overflow 방지: mask 연산은 float32로
        qv_f = qv.float()
        qm = torch.full_like(qv_f, -1e9, dtype=torch.float32)
        idx = torch.as_tensor(allowed, device=qv_f.device, dtype=torch.long)
        qm[idx] = qv_f[idx]
        a = int(torch.argmax(qm).item())
        return a, 0.0, 0.0

    def store(
        self,
        state,
        action,
        reward,
        done,
        log_prob=None,
        value=None,
        action_mask=None,
        next_state=None,
        next_action_mask=None,
    ):
        if next_state is None:
            return
        self.replay.add(
            s=state,
            a=int(action),
            r=float(reward),
            d=bool(done),
            s2=next_state,
            action_mask=action_mask,
            next_action_mask=next_action_mask,
        )
        self.global_step += 1

    def should_update(self) -> bool:
        # 최소 배치 확보
        if len(self.replay) < self.batch_size:
            return False
        # learning_starts는 env step 기준
        if self.global_step < self.learning_starts:
            return False
        # 업데이트 주기
        if (self.global_step % self.train_every_steps) != 0:
            return False
        return True

    def _batch_to_torch(self, sample) -> _ReplayBatch:
        s_u8, a, r, d, s2_u8, mask_b, next_mask_b = sample

        # uint8 -> float32 (0..1)
        s = torch.from_numpy(s_u8).to(dtype=torch.float32).mul_(1.0 / 255.0)
        s2 = torch.from_numpy(s2_u8).to(dtype=torch.float32).mul_(1.0 / 255.0)

        a_t = torch.from_numpy(a).to(dtype=torch.long)
        r_t = torch.from_numpy(r).to(dtype=torch.float32)
        d_t = torch.from_numpy(d).to(dtype=torch.float32)

        mask_t = torch.from_numpy(mask_b.astype(np.bool_, copy=False))
        next_mask_t = torch.from_numpy(next_mask_b.astype(np.bool_, copy=False))

        if self.pin_memory and self.device == "cuda":
            s = s.pin_memory()
            s2 = s2.pin_memory()
            a_t = a_t.pin_memory()
            r_t = r_t.pin_memory()
            d_t = d_t.pin_memory()
            mask_t = mask_t.pin_memory()
            next_mask_t = next_mask_t.pin_memory()

        nb = bool(self.pin_memory and self.device == "cuda")
        s = s.to(self.device, non_blocking=nb)
        s2 = s2.to(self.device, non_blocking=nb)
        a_t = a_t.to(self.device, non_blocking=nb)
        r_t = r_t.to(self.device, non_blocking=nb)
        d_t = d_t.to(self.device, non_blocking=nb)
        mask_t = mask_t.to(self.device, non_blocking=nb)
        next_mask_t = next_mask_t.to(self.device, non_blocking=nb)

        if s.is_cuda and s.dim() == 4 and self.channels_last:
            s = s.contiguous(memory_format=torch.channels_last)
            s2 = s2.contiguous(memory_format=torch.channels_last)

        return _ReplayBatch(s=s, a=a_t, r=r_t, d=d_t, s2=s2, mask=mask_t, next_mask=next_mask_t)

    def update(self, last_state=None, last_done=False):
        """
        ✅ 1 optimizer step 기준으로 update_step += 1
        ✅ grad_accum_steps 만큼 microbatch를 누적해서 GPU 연산량을 늘림
        """
        if len(self.replay) < self.batch_size:
            return None
        if self.global_step < self.learning_starts:
            return None

        self.optimizer.zero_grad(set_to_none=True)

        loss_accum = 0.0

        for k in range(self.grad_accum_steps):
            sample = self.replay.sample(self.batch_size)
            batch = self._batch_to_torch(sample)

            with torch.no_grad():
                # target Q
                q2_logits, _ = self.q_tgt(batch.s2)  # (B,A)

                if self.double_dqn:
                    q2_online, _ = self.q(batch.s2)   # (B,A)
                    q2m = q2_online.masked_fill(~batch.next_mask, -1e9)
                    a2 = torch.argmax(q2m, dim=1)     # (B,)
                    q2 = q2_logits.gather(1, a2.unsqueeze(1)).squeeze(1)
                else:
                    q2m = q2_logits.masked_fill(~batch.next_mask, -1e9)
                    q2 = torch.max(q2m, dim=1).values

                target = batch.r + (1.0 - batch.d) * self.gamma * q2

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                q_logits, _ = self.q(batch.s)  # (B,A)
                q_sa = q_logits.gather(1, batch.a.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(q_sa, target)

                # ✅ 누적이므로 평균 내기(스케일 안정)
                loss = loss / float(self.grad_accum_steps)

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_accum += float(loss.detach().item())

        # step
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.update_step += 1

        # target sync
        if (self.update_step % self.target_update_every_steps) == 0:
            self.q_tgt.load_state_dict(self.q.state_dict(), strict=False)

        return {"loss": float(loss_accum), "update_step": int(self.update_step)}

    # runner abort 호환
    def reset_buffer(self):
        return

    def clear(self):
        self.replay.clear()

    def save(self, path: str):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save(
            {
                "q": self.q.state_dict(),
                "q_tgt": self.q_tgt.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict() if self.use_amp else None,
                "global_step": int(self.global_step),
                "update_step": int(self.update_step),
                "eps_start": float(self.eps_start),
                "eps_end": float(self.eps_end),
                "eps_decay_steps": int(self.eps_decay_steps),
                "grad_accum_steps": int(self.grad_accum_steps),
            },
            path,
        )
        return True

    def load(self, path: str, load_optimizer: bool = True):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q.load_state_dict(ckpt.get("q", ckpt), strict=False)
        if "q_tgt" in ckpt:
            self.q_tgt.load_state_dict(ckpt["q_tgt"], strict=False)
        else:
            self.q_tgt.load_state_dict(self.q.state_dict(), strict=False)

        if load_optimizer and "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[WARN] optimizer state skipped: {e}")

        if self.use_amp and ("scaler" in ckpt) and (ckpt["scaler"] is not None):
            try:
                self.scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f"[WARN] scaler state skipped: {e}")

        self.global_step = int(ckpt.get("global_step", self.global_step))
        self.update_step = int(ckpt.get("update_step", self.update_step))

        self.eps_start = float(ckpt.get("eps_start", self.eps_start))
        self.eps_end = float(ckpt.get("eps_end", self.eps_end))
        self.eps_decay_steps = int(ckpt.get("eps_decay_steps", self.eps_decay_steps))

        # grad_accum_steps는 로드하되, 현재 실행 설정 우선하고 싶으면 주석 처리하면 됨
        self.grad_accum_steps = int(ckpt.get("grad_accum_steps", self.grad_accum_steps))

        return True
