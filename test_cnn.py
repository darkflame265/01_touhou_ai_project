import torch
from models.low.cnn_dqn import DQNCNN

# 행동 개수 예시 (네 actions 개수로 맞춰도 됨)
NUM_ACTIONS = 9

model = DQNCNN(num_actions=NUM_ACTIONS)

# 가짜 입력 데이터 (batch_size=1)
dummy_input = torch.zeros((1, 1, 84, 84))

output = model(dummy_input)

print("출력 shape:", output.shape)
print("출력 값:", output)
