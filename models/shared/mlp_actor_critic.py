import torch
import torch.nn as nn


class ActorCriticMLP(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_actions = int(num_actions)

        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(int(hidden_dim), self.num_actions)
        self.value_head = nn.Linear(int(hidden_dim), 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        h = self.backbone(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value
