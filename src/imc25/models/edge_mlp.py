import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeMLP(nn.Module):
    """
    Inference-only Edge MLP
    Takes two embedding tensors (zi, zj) and outputs a probability in [0, 1].
    """

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, zi, zj):
        cos = F.cosine_similarity(zi, zj).unsqueeze(1)
        x = torch.cat([zi * zj, torch.abs(zi - zj), cos], dim=1)
        return torch.sigmoid(self.net(x)).squeeze(1)
    


    
