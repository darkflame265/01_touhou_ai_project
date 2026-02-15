import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from models.shared.mlp_actor_critic import ActorCriticMLP


class MLPPPOAgent:
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        lr=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.10,
        vf_coef=0.5,
        ent_coef=0.02,
        ent_min=0.002,
        ent_decay=0.9990,
        rollout_steps=256,
        update_epochs=5,
        mini_batch_size=128,
        max_grad_norm=0.5,
        ent_warmup_updates=50,
        value_clip_eps=0.2,
        hidden_dim=256,
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = False

        self.num_actions = int(num_actions)
        self.input_dim = int(input_dim)

        self.model = ActorCriticMLP(
            input_dim=self.input_dim,
            num_actions=self.num_actions,
            hidden_dim=int(hidden_dim),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))

        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)
        self.value_clip_eps = float(value_clip_eps)

        self.ent_coef = float(ent_coef)
        self.ent_min = float(ent_min)
        self.ent_decay = float(ent_decay)
        self.ent_warmup_updates = int(ent_warmup_updates)

        self.rollout_steps = int(rollout_steps)
        self.update_epochs = int(update_epochs)
        self.mini_batch_size = int(mini_batch_size)
        self.max_grad_norm = float(max_grad_norm)

        self.global_step = 0
        self.update_step = 0

        self.last_entropy = 0.0
        self.last_kl = 0.0
        self.last_clipfrac = 0.0
        self.last_lr = float(lr)
        self.last_explained_variance = 0.0
        self.last_value_loss = 0.0
        self.last_policy_loss = 0.0
        self.last_total_loss = 0.0

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
        t = self.rollout_steps
        a = self.num_actions

        self.states = np.zeros((t,) + self._state_shape, dtype=np.float32)
        self.actions = np.zeros((t,), dtype=np.int64)
        self.rewards = np.zeros((t,), dtype=np.float32)
        self.dones = np.zeros((t,), dtype=np.float32)
        self.log_probs = np.zeros((t,), dtype=np.float32)
        self.values = np.zeros((t,), dtype=np.float32)
        self.action_masks = np.ones((t, a), dtype=np.bool_)

        self._buf_inited = True

    @staticmethod
    def _apply_action_mask_to_logits(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return logits
        mask = mask.to(dtype=torch.bool)
        row_any = mask.any(dim=-1, keepdim=True)
        safe_mask = torch.where(row_any, mask, torch.ones_like(mask, dtype=torch.bool))
        return logits.masked_fill(~safe_mask, -1e9)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, action_mask: np.ndarray | None = None, deterministic: bool = False):
        s = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device).unsqueeze(0)
        logits, value = self.model(s)

        if action_mask is not None:
            m = torch.as_tensor(action_mask, device=self.device, dtype=torch.bool).view(1, -1)
            if m.numel() == logits.size(-1):
                logits = self._apply_action_mask_to_logits(logits, m)

        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        return int(action.item()), float(dist.log_prob(action).item()), float(value.squeeze(-1).item())

    def store(self, state, action, reward, done, log_prob, value, action_mask=None):
        self._ensure_buffer(np.asarray(state, dtype=np.float32))
        i = self._buf_ptr
        if i >= self.rollout_steps:
            return

        self.states[i] = np.asarray(state, dtype=np.float32)
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
        t = rewards_t.size(0)
        adv = torch.zeros(t, device=self.device, dtype=torch.float32)
        gae = torch.zeros((), device=self.device, dtype=torch.float32)

        next_value = last_value_t
        for i in reversed(range(t)):
            mask = 1.0 - dones_t[i]
            delta = rewards_t[i] + self.gamma * next_value * mask - values_t[i]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            adv[i] = gae
            next_value = values_t[i]

        ret = adv + values_t
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        return ret, adv

    @staticmethod
    def _explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        var_y = torch.var(y_true, unbiased=False)
        if float(var_y.item()) < 1e-12:
            return 0.0
        return float((1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y).clamp(-1.0, 1.0).item())

    def _refresh_last_lr(self):
        try:
            self.last_lr = float(self.optimizer.param_groups[0].get("lr", self.last_lr))
        except Exception:
            pass

    def update(self, last_state=None, last_done=False):
        t = self._buf_ptr
        if t < 2:
            self.reset_buffer()
            self._refresh_last_lr()
            return None

        last_value = 0.0
        if last_state is not None and (not last_done):
            with torch.no_grad():
                s = torch.from_numpy(np.asarray(last_state, dtype=np.float32)).to(self.device).unsqueeze(0)
                _, v = self.model(s)
                last_value = float(v.squeeze(-1).item())

        states = torch.from_numpy(self.states[:t]).to(self.device)
        actions = torch.from_numpy(self.actions[:t]).to(self.device)
        rewards = torch.from_numpy(self.rewards[:t]).to(self.device)
        dones = torch.from_numpy(self.dones[:t]).to(self.device)
        old_log_probs = torch.from_numpy(self.log_probs[:t]).to(self.device)
        old_values = torch.from_numpy(self.values[:t]).to(self.device)
        masks = torch.from_numpy(self.action_masks[:t].astype(np.bool_, copy=False)).to(self.device)

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
                logits, v = self.model(states[mb])
                v = v.squeeze(-1)
                logits = self._apply_action_mask_to_logits(logits, masks[mb])

                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - old_log_probs[mb]
                ratio = torch.exp(log_ratio)

                surr1 = ratio * advantages[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.value_clip_eps is not None and self.value_clip_eps > 0:
                    v_old = old_values[mb]
                    v_clipped = v_old + torch.clamp(v - v_old, -self.value_clip_eps, self.value_clip_eps)
                    v_loss_1 = (v - returns[mb]).pow(2)
                    v_loss_2 = (v_clipped - returns[mb]).pow(2)
                    value_loss = 0.5 * torch.max(v_loss_1, v_loss_2).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(v, returns[mb])

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                approx_kl = 0.5 * (log_ratio.pow(2)).mean()
                clipfrac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean()

                self.optimizer.zero_grad(set_to_none=True)
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

        with torch.no_grad():
            _, v_all = self.model(states)
            ev = self._explained_variance(returns, v_all.squeeze(-1))

        self.reset_buffer()
        self._refresh_last_lr()

        if steps <= 0:
            return None

        mean_loss = total_loss / steps
        mean_policy = total_policy / steps
        mean_value = total_value / steps
        mean_entropy = total_entropy / steps
        mean_kl = total_kl / steps
        mean_clipfrac = total_clipfrac / steps

        self.last_total_loss = float(mean_loss)
        self.last_policy_loss = float(mean_policy)
        self.last_value_loss = float(mean_value)
        self.last_entropy = float(mean_entropy)
        self.last_kl = float(mean_kl)
        self.last_clipfrac = float(mean_clipfrac)
        self.last_explained_variance = float(ev)

        return {
            "loss": mean_loss,
            "policy_loss": mean_policy,
            "value_loss": mean_value,
            "entropy": mean_entropy,
            "entropy_coef": float(self.ent_coef),
            "approx_kl": mean_kl,
            "clipfrac": mean_clipfrac,
            "explained_variance": float(ev),
            "rollout_steps": int(n),
            "update_step": int(self.update_step),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "agent_type": "mlp_ppo",
                "input_dim": int(self.input_dim),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "update_step": self.update_step,
                "ent_coef": float(self.ent_coef),
                "last_entropy": float(self.last_entropy),
                "last_kl": float(self.last_kl),
                "last_clipfrac": float(self.last_clipfrac),
                "last_lr": float(self.last_lr),
                "last_explained_variance": float(self.last_explained_variance),
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        sd = ckpt.get("model", ckpt)

        # Allow architecture drift (e.g., TOP-K/input_dim changes):
        # load only keys that exist and have matching shapes.
        cur_sd = self.model.state_dict()
        filtered = {}
        skipped = []
        for k, v in sd.items():
            if (k in cur_sd) and (tuple(cur_sd[k].shape) == tuple(v.shape)):
                filtered[k] = v
            else:
                skipped.append(k)

        self.model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"[WARN] partial checkpoint load: skipped {len(skipped)} incompatible keys")

        if load_optimizer and "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[WARN] optimizer state skipped: {e}")

        self.global_step = int(ckpt.get("global_step", self.global_step))
        self.update_step = int(ckpt.get("update_step", self.update_step))
        self.ent_coef = float(ckpt.get("ent_coef", self.ent_coef))

        try:
            self.last_entropy = float(ckpt.get("last_entropy", self.last_entropy))
            self.last_kl = float(ckpt.get("last_kl", self.last_kl))
            self.last_clipfrac = float(ckpt.get("last_clipfrac", self.last_clipfrac))
            self.last_lr = float(ckpt.get("last_lr", self.last_lr))
            self.last_explained_variance = float(ckpt.get("last_explained_variance", self.last_explained_variance))
        except Exception:
            pass
