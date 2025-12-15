import numpy as np
from agents.dqn_agent import DQNAgent

NUM_ACTIONS = 9

agent = DQNAgent(num_actions=NUM_ACTIONS)

# 가짜 state (AI 시야)
dummy_state = np.zeros((84, 84), dtype=np.float32)

for i in range(10):
    action = agent.act(dummy_state)
    print("선택한 행동:", action)
