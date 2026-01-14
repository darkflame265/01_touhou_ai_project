# agents/ppo_agent.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from models.shared.cnn_actor_critic import ActorCriticCNN


class PPOAgent:
    """
    PPOAgent with action masking support.

    Improvements vs your current version:
    - Value function clipping (PPO2-style) to stabilize value learning.
    - Extra PPO diagnostics: approx_kl, clipfrac, explained_variance.
    - Safer entropy schedule defaults (less "constant shaking" in late training).
    - Minor perf: avoid re-allocating neg tensor every masking call.
    """

    def __init__(
        self,
        input_channels,
        num_actions,
        obs_channels_per_frame=4,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.15,
        vf_coef=0.5,
        # ✅ entropy: safer defaults for your task
        ent_coef=0.01,
        ent_min=0.001,
        ent_decay=0.999,
        rollout_steps=256,
        update_epochs=5,
        mini_batch_size=128,
        max_grad_norm=0.5,
        ent_warmup_updates=15,
        # ✅ value clipping
        value_clip_eps=0.2,
        device=None,
        # ---- speed options ----
        use_amp=True,
        cudnn_benchmark=True,
        compile_model=False,
        channels_last=False,
        pin_memory=True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(use_amp and (self.device == "cuda"))
        self.pin_memory = bool(pin_memory and (self.device == "cuda"))

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

        self.num_actions = int(num_actions)

        self.model = ActorCriticCNN(
            input_channels=int(input_channels),
            num_actions=int(num_actions),
            obs_channels_per_frame=int(obs_channels_per_frame),
            meta_patch=4,
            meta_channel_offset=0,
        ).to(self.device)

        self.channels_last = bool(channels_last and self.device == "cuda")
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if compile_model:
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"[WARN] torch.compile failed, continue without it: {e}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # PPO params
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)

        # Value clipping
        self.value_clip_eps = float(value_clip_eps)

        # Entropy schedule
        self.ent_coef = float(ent_coef)
        self.ent_min = float(ent_min)
        self.ent_decay = float(ent_decay)
        self.ent_warmup_updates = int(ent_warmup_updates)

        # Rollout / update
        self.rollout_steps = int(rollout_steps)
        self.update_epochs = int(update_epochs)
        self.mini_batch_size = int(mini_batch_size)
        self.max_grad_norm = float(max_grad_norm)

        self.global_step = 0
        self.update_step = 0

        # cached "very negative" for masking
        self._neg_large_fp16 = None
        self._neg_large_fp32 = None

        # ---- prealloc buffers ----
        self._buf_inited = False
        self._buf_ptr = 0
        self._state_shape = None
        self.reset_buffer()

    def reset_buffer(self):
        self._buf_ptr = 0

    def _ensure_buffer(self, state: np.ndarray):
        if self._buf_inited:
            return

        self._state_shape = tuple(state.shape)
        T = self.rollout_steps
        A = self.num_actions

        self.states = np.zeros((T,) + self._state_shape, dtype=np.float32)

        self.actions = np.zeros((T,), dtype=np.int64)
        self.rewards = np.zeros((T,), dtype=np.float32)
        self.dones = np.zeros((T,), dtype=np.float32)       # 1.0 if done else 0.0
        self.log_probs = np.zeros((T,), dtype=np.float32)
        self.values = np.zeros((T,), dtype=np.float32)

        # Action masks: bool (T, A)  True=allowed, False=forbidden
        self.action_masks = np.ones((T, A), dtype=np.bool_)

        self._buf_inited = True

    def _get_neg_large(self, dtype: torch.dtype) -> torch.Tensor:
        # allocate once per dtype/device
        if dtype in (torch.float16, torch.bfloat16):
            if self._neg_large_fp16 is None or self._neg_large_fp16.dtype != dtype or self._neg_large_fp16.device != torch.device(self.device):
                self._neg_large_fp16 = torch.tensor(-1e4, device=self.device, dtype=dtype)
            return self._neg_large_fp16
        else:
            if self._neg_large_fp32 is None or self._neg_large_fp32.dtype != dtype or self._neg_large_fp32.device != torch.device(self.device):
                self._neg_large_fp32 = torch.tensor(-1e9, device=self.device, dtype=dtype)
            return self._neg_large_fp32

    def _apply_action_mask_to_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, A)
        mask:  (B, A) bool, True=allowed
        Forbidden actions get very negative logits so prob ~ 0.
        Safety: if a row is all False, we leave logits unchanged for that row.
        """
        if mask is None:
            return logits

        mask = mask.to(dtype=torch.bool)

        row_any = mask.any(dim=-1, keepdim=True)  # (B,1)
        safe_mask = torch.where(row_any, mask, torch.ones_like(mask, dtype=torch.bool))

        neg = self._get_neg_large(logits.dtype)
        return logits.masked_fill(~safe_mask, neg)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, action_mask: np.ndarray | None = None):
        s = torch.from_numpy(state).to(self.device, dtype=torch.float32).unsqueeze(0)
        if s.is_cuda and s.dim() == 4 and self.channels_last:
            s = s.contiguous(memory_format=torch.channels_last)

        logits, value = self.model(s)

        if action_mask is not None:
            m = torch.as_tensor(action_mask, device=self.device, dtype=torch.bool).view(1, -1)
            if m.numel() == logits.size(-1):
                logits = self._apply_action_mask_to_logits(logits, m)

        dist = Categorical(logits=logits)
        action = dist.sample()

        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(value.squeeze(-1).item()),
        )

    def store(self, state, action, reward, done, log_prob, value, action_mask=None):
        self._ensure_buffer(state)

        i = self._buf_ptr
        if i >= self.rollout_steps:
            return

        self.states[i] = state.astype(np.float32, copy=False)
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.dones[i] = 1.0 if bool(done) else 0.0
        self.log_probs[i] = float(log_prob)
        self.values[i] = float(value)

        if action_mask is None:
            self.action_masks[i] = True
        else:
            am = np.asarray(action_mask, dtype=np.bool_)
            self.action_masks[i] = am if (am.shape[0] == self.num_actions) else True

        self._buf_ptr += 1
        self.global_step += 1

    def should_update(self):
        return self._buf_ptr >= self.rollout_steps

    def _compute_gae_torch(self, rewards_t, dones_t, values_t, last_value_t):
        T = rewards_t.size(0)
        adv = torch.zeros(T, device=self.device, dtype=torch.float32)
        gae = torch.zeros((), device=self.device, dtype=torch.float32)

        next_value = last_value_t
        for t in reversed(range(T)):
            mask = 1.0 - dones_t[t]
            delta = rewards_t[t] + self.gamma * next_value * mask - values_t[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            adv[t] = gae
            next_value = values_t[t]

        ret = adv + values_t
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        return ret, adv

    @staticmethod
    def _explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        # 1 - Var[y_true - y_pred] / Var[y_true]
        y_true = y_true.detach()
        y_pred = y_pred.detach()
        var_y = torch.var(y_true, unbiased=False)
        if float(var_y.item()) < 1e-12:
            return 0.0
        return float((1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y).clamp(-1.0, 1.0).item())

    def update(self, last_state=None, last_done=False):
        T = self._buf_ptr
        if T < 2:
            self.reset_buffer()
            return None

        # ---- bootstrap last value ----
        last_value = 0.0
        if last_state is not None and (not last_done):
            with torch.no_grad():
                s = torch.from_numpy(last_state).to(self.device, dtype=torch.float32).unsqueeze(0)
                if s.is_cuda and s.dim() == 4 and self.channels_last:
                    s = s.contiguous(memory_format=torch.channels_last)
                _, v = self.model(s)
                last_value = float(v.squeeze(-1).item())

        # ---- slice only used part (T) ----
        states_np = self.states[:T]
        actions_np = self.actions[:T]
        rewards_np = self.rewards[:T]
        dones_np = self.dones[:T]
        old_logp_np = self.log_probs[:T]
        values_np = self.values[:T]
        masks_np = self.action_masks[:T]

        # ---- CPU tensors ----
        states_cpu = torch.from_numpy(states_np)
        actions_cpu = torch.from_numpy(actions_np)
        rewards_cpu = torch.from_numpy(rewards_np)
        dones_cpu = torch.from_numpy(dones_np)
        old_logp_cpu = torch.from_numpy(old_logp_np)
        values_cpu = torch.from_numpy(values_np)
        masks_cpu = torch.from_numpy(masks_np.astype(np.bool_, copy=False))

        if self.pin_memory:
            states_cpu = states_cpu.pin_memory()
            actions_cpu = actions_cpu.pin_memory()
            rewards_cpu = rewards_cpu.pin_memory()
            dones_cpu = dones_cpu.pin_memory()
            old_logp_cpu = old_logp_cpu.pin_memory()
            values_cpu = values_cpu.pin_memory()
            masks_cpu = masks_cpu.pin_memory()

        nb = self.pin_memory

        # ---- GPU tensors ----
        states = states_cpu.to(self.device, non_blocking=nb)
        actions = actions_cpu.to(self.device, non_blocking=nb)
        rewards = rewards_cpu.to(self.device, non_blocking=nb)
        dones = dones_cpu.to(self.device, non_blocking=nb)
        old_log_probs = old_logp_cpu.to(self.device, non_blocking=nb)
        old_values = values_cpu.to(self.device, non_blocking=nb)
        masks = masks_cpu.to(self.device, non_blocking=nb)

        if states.is_cuda and states.dim() == 4 and self.channels_last:
            states = states.contiguous(memory_format=torch.channels_last)

        last_value_t = torch.tensor(last_value, device=self.device, dtype=torch.float32)
        returns, advantages = self._compute_gae_torch(rewards, dones, old_values, last_value_t)

        n = states.size(0)

        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clipfrac = 0.0
        steps = 0

        for _ in range(self.update_epochs):
            idxs = torch.randperm(n, device=self.device)

            for start in range(0, n, self.mini_batch_size):
                mb = idxs[start:start + self.mini_batch_size]

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits, v = self.model(states[mb])
                    v = v.squeeze(-1)

                    logits = self._apply_action_mask_to_logits(logits, masks[mb])

                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions[mb])
                    entropy = dist.entropy().mean()

                    # ---- PPO policy loss ----
                    log_ratio = new_log_probs - old_log_probs[mb]
                    ratio = torch.exp(log_ratio)

                    surr1 = ratio * advantages[mb]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[mb]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # ---- value loss with clipping ----
                    if self.value_clip_eps is not None and self.value_clip_eps > 0:
                        v_old = old_values[mb]
                        v_clipped = v_old + torch.clamp(v - v_old, -self.value_clip_eps, self.value_clip_eps)
                        v_loss_1 = (v - returns[mb]).pow(2)
                        v_loss_2 = (v_clipped - returns[mb]).pow(2)
                        value_loss = 0.5 * torch.max(v_loss_1, v_loss_2).mean()
                    else:
                        value_loss = 0.5 * F.mse_loss(v, returns[mb])

                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                    # ---- diagnostics ----
                    approx_kl = 0.5 * (log_ratio.pow(2)).mean()
                    clipfrac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean()

                self.optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                total_loss += float(loss.item())
                total_policy += float(policy_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_kl += float(approx_kl.item())
                total_clipfrac += float(clipfrac.item())
                steps += 1

        self.update_step += 1
        if self.update_step > self.ent_warmup_updates:
            self.ent_coef = max(self.ent_min, self.ent_coef * self.ent_decay)

        # explained variance (full batch)
        with torch.no_grad():
            # recompute v on full states for EV
            logits_all, v_all = self.model(states)
            ev = self._explained_variance(returns, v_all.squeeze(-1))

        self.reset_buffer()

        if steps == 0:
            return None

        return {
            "loss": total_loss / steps,
            "policy_loss": total_policy / steps,
            "value_loss": total_value / steps,
            "entropy": total_entropy / steps,
            "entropy_coef": float(self.ent_coef),
            "approx_kl": total_kl / steps,
            "clipfrac": total_clipfrac / steps,
            "explained_variance": float(ev),
            "rollout_steps": int(n),
            "update_step": int(self.update_step),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict() if self.use_amp else None,
                "global_step": self.global_step,
                "update_step": self.update_step,
                "ent_coef": float(self.ent_coef),
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        sd = ckpt.get("model", ckpt)
        self.model.load_state_dict(sd, strict=False)

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
        self.ent_coef = float(ckpt.get("ent_coef", self.ent_coef))
