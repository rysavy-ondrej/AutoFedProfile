import torch
import torch.nn.functional as F

@torch.no_grad()
def build_prototypes(encoder, x_ref, y_ref, device="cpu"):
    encoder.eval()
    x_ref = x_ref.to(device)
    y_ref = y_ref.to(device)

    z_ref = encoder(x_ref)
    classes = torch.unique(y_ref)

    prototypes = {}
    for c in classes:
        z_c = z_ref[y_ref == c]
        proto = z_c.mean(dim=0, keepdim=True)
        proto = F.normalize(proto, dim=1)
        prototypes[int(c.item())] = proto.squeeze(0)

    return prototypes


@torch.no_grad()
def predict_with_prototypes(encoder, prototypes, x, device="cpu"):
    encoder.eval()
    z = encoder(x.to(device))   # [N, D]

    class_ids = sorted(prototypes.keys())
    proto_mat = torch.stack([prototypes[c] for c in class_ids], dim=0).to(device)  # [C, D]

    sims = z @ proto_mat.T   # cosine similarity because normalized embeddings
    pred_idx = sims.argmax(dim=1)

    preds = torch.tensor([class_ids[i] for i in pred_idx.tolist()])
    return preds, sims.cpu()