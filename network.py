import torch
import torch.nn as nn
import torch.nn.functional as F


class ConnectFourNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(42, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.policy_head = nn.Linear(128, 7)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value
