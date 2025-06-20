import torch.nn as nn

class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),      nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.layers(x)