import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from models.shared.cnn_actor_critic import ActorCriticCNN


class PPOAgent:
    """
    Drop-in replacement for your working PPOAgent, optimized for:
    - fewer Python list ops (preallocated rollout buffer)
    - fewer CPU->GPU transfers (batch transfer once per update)
    - faster training via AMP (fp16) on RTX 3060 Ti
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
        ent_coef=0.04,
        ent_min=0.005,
        ent_decay=0.9995,
        rollout_steps=128,
        update_epochs=5,
        mini_batch_size=64,
        max_grad_norm=0.5,
        ent_warmup_updates=30,
        device=None,
        # ---- speed options ----
        use_amp=True,              # mixed precision (recommended on 3060Ti)
        cudnn_benchmark=True,      # good when input size is fixed
        compile_model=False,       # torch.compile (PyTorch 2.x). Try if stable.
        channels_last=False,       # enable if your CNN benefits
        pin_memory=True,           # helps H2D copy speed
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(use_amp and (self.device == "cuda"))
        self.pin_memory = bool(pin_memory and (self.device == "cuda"))

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

        self.model = ActorCriticCNN(
            input_channels=int(input_channels),
            num_actions=int(num_actions),
            obs_channels_per_frame=int(obs_channels_per_frame),
            meta_patch=4,
            meta_channel_offset=0,
        ).to(self.device)

        if channels_last and self.device == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)

        if compile_model:
            try:
                self.model = torch.compile(self.model)  # PyTorch 2.x
            except Exception as e:
                print(f"[WARN] torch.compile failed, continue without it: {e}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # PPO params
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)

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

        # state expected shape: (C,H,W) or whatever your model expects
        self._state_shape = tuple(state.shape)
        T = self.rollout_steps

        # States: float32
        self.states = np.zeros((T,) + self._state_shape, dtype=np.float32)
        # Scalars:
        self.actions = np.zeros((T,), dtype=np.int64)
        self.rewards = np.zeros((T,), dtype=np.float32)
        self.dones = np.zeros((T,), dtype=np.float32)       # 1.0 if done else 0.0
        self.log_probs = np.zeros((T,), dtype=np.float32)
        self.values = np.zeros((T,), dtype=np.float32)

        self._buf_inited = True

    @torch.no_grad()
    def select_action(self, state: np.ndarray):
        # state -> (1, ...) float32 on device
        s = torch.from_numpy(state).to(self.device, dtype=torch.float32).unsqueeze(0)

        if s.is_cuda and getattr(self.model, "to", None) and s.dim() == 4:
            # If you used channels_last, keep input consistent
            # (safe even if channels_last=False)
            s = s.contiguous(memory_format=torch.channels_last)

        logits, value = self.model(s)
        dist = Categorical(logits=logits)
        action = dist.sample()

        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(value.squeeze(-1).item()),
        )

    def store(self, state, action, reward, done, log_prob, value):
        self._ensure_buffer(state)

        i = self._buf_ptr
        if i >= self.rollout_steps:
            # if user accidentally keeps storing, just ignore extra
            return

        self.states[i] = state.astype(np.float32, copy=False)
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.dones[i] = 1.0 if bool(done) else 0.0
        self.log_probs[i] = float(log_prob)
        self.values[i] = float(value)

        self._buf_ptr += 1
        self.global_step += 1

    def should_update(self):
        return self._buf_ptr >= self.rollout_steps

    def _compute_gae_torch(self, rewards_t, dones_t, values_t, last_value_t):
        """
        rewards_t: (T,)
        dones_t: (T,) float32 1.0 done else 0.0
        values_t: (T,) predicted V(s_t)
        last_value_t: scalar tensor V(s_{T}) if not done else 0
        """
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
                if s.is_cuda and s.dim() == 4:
                    s = s.contiguous(memory_format=torch.channels_last)
                _, v = self.model(s)
                last_value = float(v.squeeze(-1).item())

        # ---- batch tensors (one transfer) ----
        # Slice only used part (T)
        states_np = self.states[:T]
        actions_np = self.actions[:T]
        rewards_np = self.rewards[:T]
        dones_np = self.dones[:T]
        old_logp_np = self.log_probs[:T]
        values_np = self.values[:T]

        # CPU tensors
        states_cpu = torch.from_numpy(states_np)  # float32
        actions_cpu = torch.from_numpy(actions_np)  # int64
        rewards_cpu = torch.from_numpy(rewards_np)  # float32
        dones_cpu = torch.from_numpy(dones_np)      # float32
        old_logp_cpu = torch.from_numpy(old_logp_np)  # float32
        values_cpu = torch.from_numpy(values_np)    # float32

        if self.pin_memory:
            states_cpu = states_cpu.pin_memory()
            actions_cpu = actions_cpu.pin_memory()
            rewards_cpu = rewards_cpu.pin_memory()
            dones_cpu = dones_cpu.pin_memory()
            old_logp_cpu = old_logp_cpu.pin_memory()
            values_cpu = values_cpu.pin_memory()

        # GPU tensors (non_blocking if pinned)
        nb = self.pin_memory
        states = states_cpu.to(self.device, non_blocking=nb)
        actions = actions_cpu.to(self.device, non_blocking=nb)
        rewards = rewards_cpu.to(self.device, non_blocking=nb)
        dones = dones_cpu.to(self.device, non_blocking=nb)
        old_log_probs = old_logp_cpu.to(self.device, non_blocking=nb)
        values = values_cpu.to(self.device, non_blocking=nb)

        if states.is_cuda and states.dim() == 4:
            states = states.contiguous(memory_format=torch.channels_last)

        last_value_t = torch.tensor(last_value, device=self.device, dtype=torch.float32)
        returns, advantages = self._compute_gae_torch(rewards, dones, values, last_value_t)

        n = states.size(0)
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        steps = 0

        for _ in range(self.update_epochs):
            idxs = torch.randperm(n, device=self.device)

            for start in range(0, n, self.mini_batch_size):
                mb = idxs[start:start + self.mini_batch_size]

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits, v = self.model(states[mb])
                    v = v.squeeze(-1)

                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions[mb])
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - old_log_probs[mb])
                    surr1 = ratio * advantages[mb]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[mb]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(v, returns[mb])
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

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
                steps += 1

        self.update_step += 1
        if self.update_step > self.ent_warmup_updates:
            self.ent_coef = max(self.ent_min, self.ent_coef * self.ent_decay)

        self.reset_buffer()

        if steps == 0:
            return None

        return {
            "loss": total_loss / steps,
            "policy_loss": total_policy / steps,
            "value_loss": total_value / steps,
            "entropy": total_entropy / steps,
            "entropy_coef": float(self.ent_coef),
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
