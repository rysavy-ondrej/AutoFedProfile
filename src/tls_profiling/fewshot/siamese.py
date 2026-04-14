import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .encoders import build_encoder


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)
# ============================================================
# 2. Pair dataset for Siamese training
# ============================================================

class SiamesePairDataset(Dataset):
    """
    Creates positive and negative pairs from labeled data.
    label = 1.0 means same class
    label = 0.0 means different class
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, pairs_per_epoch: int = 5000):
        self.x = x
        self.y = y
        self.pairs_per_epoch = pairs_per_epoch

        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(y.tolist()):
            self.class_to_indices.setdefault(label, []).append(idx)

        self.classes = sorted(self.class_to_indices.keys())

        if len(self.classes) < 2:
            raise ValueError("At least two classes are required for Siamese training.")

        for cls, indices in self.class_to_indices.items():
            if len(indices) < 2:
                raise ValueError(f"Class {cls} must contain at least two samples.")

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def __getitem__(self, index: int):
        is_positive = random.random() < 0.5

        if is_positive:
            cls = random.choice(self.classes)
            idx1, idx2 = random.sample(self.class_to_indices[cls], 2)
            label = torch.tensor(1.0, dtype=torch.float32)
        else:
            cls1, cls2 = random.sample(self.classes, 2)
            idx1 = random.choice(self.class_to_indices[cls1])
            idx2 = random.choice(self.class_to_indices[cls2])
            label = torch.tensor(0.0, dtype=torch.float32)

        return self.x[idx1], self.x[idx2], label


class SiameseNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        emb_dim: int = 32,
        encoder: Union[str, nn.Module] = "default",
    ):
        super().__init__()
        self.encoder = build_encoder(encoder=encoder, input_dim=input_dim, emb_dim=emb_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        return z1, z2


# ============================================================
# 4. Contrastive loss
# ============================================================

def contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    y: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    y = 1 -> same class
    y = 0 -> different class
    """
    cosine_sim = F.cosine_similarity(z1, z2)
    dist = 1.0 - cosine_sim

    loss_pos = y * dist.pow(2)
    loss_neg = (1.0 - y) * torch.clamp(margin - dist, min=0.0).pow(2)
    return (loss_pos + loss_neg).mean()


# ============================================================
# 5. Training config
# ============================================================

@dataclass
class TrainConfig:
    input_dim: int
    emb_dim: int = 32
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    margin: float = 1.0
    pairs_per_epoch: int = 6000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 6. Training
# ============================================================

def train_siamese(
    model: SiameseNetwork,
    train_loader: DataLoader,
    cfg: TrainConfig,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()

    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        for x1, x2, y in train_loader:
            x1 = x1.to(cfg.device)
            x2 = x2.to(cfg.device)
            y = y.to(cfg.device)

            z1, z2 = model(x1, x2)
            loss = contrastive_loss(z1, z2, y, margin=cfg.margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch:02d} | loss = {epoch_loss / max(n_batches, 1):.4f}")


# ============================================================
# 7. Few-shot episode sampling
# ============================================================

def sample_few_shot_episode(
    x: torch.Tensor,
    y: torch.Tensor,
    n_way: int,
    k_shot: int,
    q_query: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    classes = sorted(set(y.tolist()))
    if len(classes) < n_way:
        raise ValueError(f"Need at least {n_way} classes, got {len(classes)}.")

    selected_classes = random.sample(classes, n_way)

    support_x_list = []
    support_y_list = []
    query_x_list = []
    query_y_list = []

    for new_label, cls in enumerate(selected_classes):
        indices = (y == cls).nonzero(as_tuple=True)[0].tolist()

        if len(indices) < k_shot + q_query:
            raise ValueError(
                f"Class {cls} has only {len(indices)} samples, "
                f"but {k_shot + q_query} are needed."
            )

        chosen = random.sample(indices, k_shot + q_query)
        support_idx = chosen[:k_shot]
        query_idx = chosen[k_shot:]

        support_x_list.append(x[support_idx])
        support_y_list.append(torch.full((k_shot,), new_label, dtype=torch.long))

        query_x_list.append(x[query_idx])
        query_y_list.append(torch.full((q_query,), new_label, dtype=torch.long))

    support_x = torch.cat(support_x_list, dim=0)
    support_y = torch.cat(support_y_list, dim=0)
    query_x = torch.cat(query_x_list, dim=0)
    query_y = torch.cat(query_y_list, dim=0)

    return support_x, support_y, query_x, query_y


# ============================================================
# 8. Few-shot inference using prototypes
# ============================================================

@torch.no_grad()
def compute_prototypes(
    encoder: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    n_way: int,
    device: str,
) -> torch.Tensor:
    encoder.eval()
    z_support = encoder(support_x.to(device))

    prototypes = []
    support_y = support_y.to(device)

    for c in range(n_way):
        z_c = z_support[support_y == c]
        proto = z_c.mean(dim=0)
        proto = F.normalize(proto.unsqueeze(0), dim=1).squeeze(0)
        prototypes.append(proto)

    return torch.stack(prototypes, dim=0)


@torch.no_grad()
def classify_queries(
    encoder: nn.Module,
    prototypes: torch.Tensor,
    query_x: torch.Tensor,
    device: str,
) -> torch.Tensor:
    encoder.eval()
    z_query = encoder(query_x.to(device))
    sims = z_query @ prototypes.T
    preds = sims.argmax(dim=1)
    return preds.cpu()


@torch.no_grad()
def evaluate_few_shot(
    encoder: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    n_way: int,
    k_shot: int,
    q_query: int,
    n_episodes: int,
    device: str,
) -> float:
    accs = []

    for _ in range(n_episodes):
        support_x, support_y, query_x, query_y = sample_few_shot_episode(
            x=x_test,
            y=y_test,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
        )

        prototypes = compute_prototypes(
            encoder=encoder,
            support_x=support_x,
            support_y=support_y,
            n_way=n_way,
            device=device,
        )

        preds = classify_queries(
            encoder=encoder,
            prototypes=prototypes,
            query_x=query_x,
            device=device,
        )

        acc = (preds == query_y).float().mean().item()
        accs.append(acc)

    return sum(accs) / len(accs)


# ============================================================
# 9. Main externally-driven pipeline
# ============================================================

def run_siamese_few_shot(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    input_dim: int,
    emb_dim: int = 32,
    batch_size: int = 128,
    epochs: int = 20,
    lr: float = 1e-3,
    margin: float = 1.0,
    pairs_per_epoch: int = 6000,
    n_way: int = 2,
    k_shot_values=(1, 5),
    q_query: int = 10,
    n_episodes: int = 100,
    device: str|None = None,
    encoder: Union[str, nn.Module] = "default",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = TrainConfig(
        input_dim=input_dim,
        emb_dim=emb_dim,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        margin=margin,
        pairs_per_epoch=pairs_per_epoch,
        device=device,
    )

    # Ensure correct dtype
    x_train = x_train.float()
    x_test = x_test.float()
    y_train = y_train.long()
    y_test = y_test.long()

    if x_train.shape[1] != input_dim:
        raise ValueError(f"x_train has shape {x_train.shape}, expected second dimension {input_dim}.")
    if x_test.shape[1] != input_dim:
        raise ValueError(f"x_test has shape {x_test.shape}, expected second dimension {input_dim}.")

    pair_dataset = SiamesePairDataset(
        x=x_train,
        y=y_train,
        pairs_per_epoch=cfg.pairs_per_epoch,
    )

    train_loader = DataLoader(
        pair_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = SiameseNetwork(
        input_dim=input_dim,
        emb_dim=emb_dim,
        encoder=encoder,
    ).to(cfg.device)

    print(model)
    train_siamese(model=model, train_loader=train_loader, cfg=cfg)

    results = {}
    for k_shot in k_shot_values:
        acc = evaluate_few_shot(
            encoder=model.encoder,
            x_test=x_test,
            y_test=y_test,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
            n_episodes=n_episodes,
            device=cfg.device,
        )
        results[k_shot] = acc
        print(f"Few-shot accuracy | {n_way}-way {k_shot}-shot: {acc:.4f}")

    return model, results
