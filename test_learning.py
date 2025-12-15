import numpy as np
from agents.dqn_agent import DQNAgent

NUM_ACTIONS = 9

agent = DQNAgent(num_actions=NUM_ACTIONS)

# 가짜 데이터 100개 넣기
for _ in range(100):
    s = np.zeros((84, 84), dtype=np.float32)
    a = np.random.randint(NUM_ACTIONS)
    r = np.random.randn()
    ns = np.zeros((84, 84), dtype=np.float32)
    d = np.random.choice([0, 1])

    agent.memory.push(s, a, r, ns, d)

# 학습 시도
for i in range(10):
    agent.learn()
    print("학습 step", i)
