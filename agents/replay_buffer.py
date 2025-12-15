# agents/replay_buffer.py
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

        # 🔥 중요 경험은 추가로 더 넣어준다 (우선순위 효과)
        if reward <= -10 or done:
            for _ in range(2):   # ← 가중치 (2~4 추천)
                self.buffer.append(transition)


    def sample(self, batch_size):
        batch_size = int(batch_size)
        return random.sample(self.buffer, batch_size)

    # ✅ 저장용: 리스트로 변환
    def state_dict(self):
        return {
            "capacity": self.capacity,
            "buffer": list(self.buffer),
        }

    # ✅ 복원용
    def load_state_dict(self, d):
        self.capacity = int(d.get("capacity", self.capacity))
        self.buffer = deque(d.get("buffer", []), maxlen=self.capacity)
