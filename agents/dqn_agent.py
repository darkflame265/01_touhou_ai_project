# agents/dqn_agent.py
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy

from agents.replay_buffer import ReplayBuffer
from models.low.cnn_dqn import DQNCNN


class DQNAgent:
    def __init__(
        self,
        input_channels,
        num_actions,
        lr=1e-4,
        gamma=0.99,
        batch_size=32,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        device=None,
        target_update_freq=1000,
        grad_clip=10.0,
        memory_capacity=100_000,
    ):
        self.device = device or torch.device("cpu")

        self.model = DQNCNN(input_channels, num_actions).to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.target_model.eval()

        self.target_update_freq = target_update_freq
        self.learn_steps = 0

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=memory_capacity)

        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.num_actions = num_actions
        self.grad_clip = float(grad_clip)

        self.last_loss = None

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        # ✅ state는 항상 float32로
        state = np.asarray(state, dtype=np.float32)
        state_t = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_t)

        return int(q_values.argmax(dim=1).item())

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)  # 0.0/1.0
        actions = np.array(actions, dtype=np.int64)

        states_t = torch.from_numpy(states).unsqueeze(1).to(self.device)       # (B,1,84,84)
        next_states_t = torch.from_numpy(next_states).unsqueeze(1).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)                  # (B,)
        rewards_t = torch.from_numpy(rewards).to(self.device)                  # (B,)
        dones_t = torch.from_numpy(dones).to(self.device)                      # (B,)

        q_values = self.model(states_t)
        q_value = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_model(next_states_t).max(1)[0]

        target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = F.mse_loss(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print("[DEBUG] target network updated")

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.last_loss = float(loss.item())
        return self.last_loss

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_steps": self.learn_steps,
        }, path)

    def load(self, path, load_optimizer=True):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.target_model.load_state_dict(ckpt.get("target_model", ckpt["model"]))
        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = float(ckpt.get("epsilon", self.epsilon))
        self.learn_steps = int(ckpt.get("learn_steps", self.learn_steps))
        self.target_model.eval()
