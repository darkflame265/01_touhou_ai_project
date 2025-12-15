import torch
import torch.nn as nn


class DQNCNN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )

        # ✅ 해상도 무관 핵심
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = self.pool(x)                 # 🔥 이 줄이 핵심
        x = x.view(x.size(0), -1)
        return self.fc(x)
