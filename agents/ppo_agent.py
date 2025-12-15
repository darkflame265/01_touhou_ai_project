import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

from models.shared.cnn_actor_critic import ActorCriticCNN


class PPOAgent:
    def __init__(
        self,
        input_channels,
        num_actions,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device=None,
    ):
        self.device = device or torch.device("cpu")

        self.model = ActorCriticCNN(input_channels, num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.reset_buffer()

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def select_action(self, state):
        state = np.asarray(state, dtype=np.float32)
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)

        logits, value = self.model(state_t)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return (
            int(action.item()),
            log_prob.item(),
            value.item(),
        )

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def finish_episode(self):
        # GAE 계산
        returns = []
        advantages = []

        last_value = 0.0
        gae = 0.0

        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + self.gamma * last_value * mask
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
            last_value = self.values[t]

        # Tensor 변환
        states = torch.from_numpy(np.array(self.states)).to(self.device)
        actions = torch.tensor(self.actions).to(self.device)
        old_log_probs = torch.tensor(self.log_probs).to(self.device)
        returns = torch.tensor(returns).to(self.device)
        advantages = torch.tensor(advantages).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO 업데이트 (여기서는 1 epoch만)
        logits, values = self.model(states)
        dist = Categorical(logits=logits)

        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reset_buffer()
