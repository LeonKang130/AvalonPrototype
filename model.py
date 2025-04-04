import torch
from torch import nn

class Lancelot(nn.Module):
    def __init__(self):
        super(Lancelot, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 26),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        lc = self.encoder(x[..., :30])
        return self.decoder(torch.hstack([lc, x[..., 30:]]))