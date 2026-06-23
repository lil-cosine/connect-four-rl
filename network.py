import torch
import torch.nn as nn
import torch.nn.functional as F


class ConnectFourNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(42, 64)
        self.fc2 = nn.Linear(64, 64)
        self.policy_head = nn.Linear(64, 42)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value
