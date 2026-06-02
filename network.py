import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block: two conv layers with a skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ConnectFourNet(nn.Module):
    """
    CNN-based policy-value network for Connect Four.

    Input:  (batch, 3, 6, 7) tensor
              channel 0 — current player's pieces  (1.0)
              channel 1 — opponent's pieces         (1.0)
              channel 2 — all-ones bias plane

    Output: policy logits (7,), value scalar in [-1, 1]
    """
    def __init__(self, channels=128, num_res_blocks=6):
        super().__init__()

        # Entry convolution: project 3 input planes → `channels` feature maps
        self.entry = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # Residual tower
        self.res_tower = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res_blocks)]
        )

        # Policy head: deeper than value head for better move ranking
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.policy_fc1 = nn.Linear(32 * 6 * 7, 128)
        self.policy_fc2 = nn.Linear(128, 7)

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.value_fc1 = nn.Linear(16 * 6 * 7, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x may arrive as (3, 6, 7) from a single state — add batch dim
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.entry(x)
        x = self.res_tower(x)

        # Policy
        p = self.policy_conv(x).flatten(1)
        p = F.relu(self.policy_fc1(p))
        policy = self.policy_fc2(p).squeeze(0)          # → (7,)

        # Value
        v = self.value_conv(x).flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(0) # → scalar

        return policy, value
