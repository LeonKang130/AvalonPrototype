import torch
from torch import nn

class Lancelot(nn.Module):
    def __init__(self):
        super(Lancelot, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        lc = self.encoder(x[..., :30])
        t, n = lc[..., -6:-3], lc[..., -3:]
        t = t - (t * n).sum(dim=-1).unsqueeze(-1) * n
        b = torch.linalg.cross(t, n)
        onb = torch.stack([t, b, n], dim=-2)
        wi, wo = x[..., -6:-3], x[..., -3:]
        wi_local = (onb.transpose(-2, -1) @ wi.unsqueeze(-1)).squeeze(-1)
        wo_local = (onb.transpose(-2, -1) @ wo.unsqueeze(-1)).squeeze(-1)
        return self.decoder(torch.hstack([lc[..., :-6], wi_local, wo_local]))