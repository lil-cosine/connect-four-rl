import torch
import torch.nn as nn
import numpy as np

class ConnectFourNet(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = nn.Sequential(
            nn.Linear(42, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        ).to(self.device)

    def forward(self, x):
        return self.net(x)

    def choose_col(self, board, player):
        state = board.get_state(player)

        x = torch.tensor(state, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            scores = self.forward(x)

        valid_cols = board.get_valid_cols()
        mask = torch.full((7,), float('-inf')).to(self.device)
        for col in valid_cols:
            mask[col] = 0.0
        scores = scores + mask

        return torch.argmax(scores).item()

    def get_weights(self):
        return np.concatenate([
            p.data.detach().cpu().numpy().flatten()
            for p in self.parameters()
        ])
    def set_weights(self, flat_weights):
        idx = 0
        for p in self.parameters():
            size = p.data.numel()
            p.data = torch.tensor(
                flat_weights[idx:idx + size],
                dtype=torch.float32
            ).reshape(p.data.shape).to(self.device)
            idx += size
