import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticCNN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )

        # 🔥 해상도 무관 핵심
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        # 정책 헤드 (행동 확률)
        self.policy_head = nn.Linear(512, num_actions)

        # 가치 헤드 (V(s))
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        logits = self.policy_head(x)
        value = self.value_head(x)

        return logits, value
