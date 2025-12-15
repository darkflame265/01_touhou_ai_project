# agents/replay_buffer.py
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # 항상 5개만 저장
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # "transition 리스트" (길이=batch_size) 반환
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
