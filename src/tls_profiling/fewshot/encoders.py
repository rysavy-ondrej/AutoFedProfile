from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1)


class CompactEncoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1)


class DeepEncoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1)


def build_encoder(
    encoder: Union[str, nn.Module],
    input_dim: int,
    emb_dim: int,
) -> nn.Module:
    if isinstance(encoder, nn.Module):
        return encoder

    encoder_builders = {
        "default": Encoder,
        "compact": CompactEncoder,
        "deep": DeepEncoder,
    }

    if encoder not in encoder_builders:
        available = ", ".join(sorted(encoder_builders))
        raise ValueError(
            f"Unknown encoder variant '{encoder}'. Available variants: {available}."
        )

    return encoder_builders[encoder](input_dim=input_dim, emb_dim=emb_dim)
