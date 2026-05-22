import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, f=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1), nn.BatchNorm2d(f, track_running_stats=False), nn.ReLU(),
            nn.Conv2d(f, f, 3, padding=1), nn.BatchNorm2d(f, track_running_stats=False)
        )

    def forward(self, x):
        return F.relu(x + self.net(x))

class ActorCriticNet(nn.Module):
    def __init__(self, num_res=4, f=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, f, 3, padding=1), nn.BatchNorm2d(f, track_running_stats=False), nn.ReLU()
        )
        self.body = nn.Sequential(*[ResBlock(f) for _ in range(num_res)])

        self.policy = nn.Sequential(
            nn.Conv2d(f, 2, 1), nn.BatchNorm2d(2, track_running_stats=False), nn.ReLU(),
            nn.Flatten(), nn.Linear(2*6*7, 7)
        )
        self.value = nn.Sequential(
            nn.Conv2d(f, 1, 1), nn.BatchNorm2d(1, track_running_stats=False), nn.ReLU(),
            nn.Flatten(), nn.Linear(6*7, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, 2, 6, 7)
        x = self.body(self.stem(x))
        return self.policy(x), self.value(x).squeeze(-1).squeeze(-1)
