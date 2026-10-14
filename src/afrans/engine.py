"""
Train/val loop and checkpointing for a single model + config pair
"""

import time
import torch
from afrans.config import CKPT_DIR

def resolve_device(train_cfg):
    if train_cfg.device != "auto":
        return torch.device(train_cfg.device)
    if torch.backends.mps.is_available():
        return torch.device("mps") # Apple GPU
    if torch.cuda.is_available():
        return torch.device("cuda") # NVIDIA GPU
    return torch.device("cpu")

def clip_grad_norm_(parameters, max_norm):
    """
    torch.nn.utils.clip_grad_norm_ calls torch.linalg.vector_norm
    directly on each gradient, which MPS doesn't support for complex
    dtypes (the FNO's spectral conv weights are complex64) 
    Works fine on CPU, MPS is the gap. Taking .abs() first sidesteps
    it and gives the identical L2 norm, since |a+bi|^2 = a^2+b^2 is 
    exactly what the norm needs.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    norms = [torch.linalg.vector_norm(g.abs() if torch.is_complex(g) else g) for g in grads]
    total_norm = torch.linalg.vector_norm(torch.stack(norms))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
             g.mul_(clip_coef)

    return total_norm

def weighted_mse(pred, target, weight):
    """
    weight: (H, W) mean 1.0
    Broadcasts agains (B, C, H, W) without reshape
    """
    return (((pred-target) ** 2) * weight).mean()

def train_one_epoch(model, loader, opt, device, weight, grad_clip=0.0):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = weighted_mse(model(x), y, weight)
        loss.backward()
        if grad_clip:
            clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n

def train_one_epoch_gnn(model, loader, opt, device, grad_clip=0.0):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = torch.nn.functional.mse_loss(out, batch.y)
        loss.backward()
        if grad_clip:
            clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total += loss.item() * batch.num_graphs
        n += batch.num_graphs

    return total/n


@torch.no_grad()
def validate(model, loader, device, weight):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = weighted_mse(model(x), y, weight=weight)
        total += loss.item() *x.size(0)
        n += x.size(0)
    return total / n

@torch.no_grad()
def validate_gnn(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        loss = torch.nn.functional.mse_loss(out, batch.y)
        total += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total/n

def checkpoint_path(train_cfg, grid_cfg):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    return CKPT_DIR / f"{train_cfg.run_name(grid_cfg)}.pt"

def fit(model, train_loader, val_loader, train_cfg, grid_cfg, weight, verbose=True):
    """
    weight: (H, W) array from cache.get_weight_map
    """
    device = resolve_device(train_cfg)
    model.to(device)
    weight_t = torch.from_numpy(weight).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr,
                           weight_decay=train_cfg.weight_decay)
    path = checkpoint_path(train_cfg, grid_cfg)

    history = {"train":[], "val": []}
    best_val = float("inf")

    for epoch in range(train_cfg.epochs):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, opt, device, weight_t, train_cfg.grad_clip)
        val_loss = validate(model, val_loader, device, weight_t)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model":model.state_dict(),
                "optimizer":opt.state_dict(),
                "epoch":epoch,
                "val_loss":val_loss,
                "train_cfg":train_cfg.to_dict(),
                "grid_cfg":grid_cfg.to_dict()
            }, path)

        if verbose:
            dt = time.perf_counter() - t0
            print(f"epoch {epoch:3d} train {train_loss:.5f} val {val_loss:.5f} "
                  f"({dt:.1f}s) {'*' if val_loss == best_val else ''}")

    return history, path

def fit_gnn(model, train_loader, val_loader, train_cfg, verbose=True):
    """
    No grid_cfg and weight map, mirrors fit() structure otherwise.
    """
    device = resolve_device(train_cfg)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr,
                           weight_decay=train_cfg.weight_decay)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{train_cfg.run_name()}.pt"

    history = {"train":[], "val":[]}
    best_val = float("inf")

    for epoch in range(train_cfg.epochs):
        t0 = time.perf_counter()
        train_loss = train_one_epoch_gnn(model, train_loader, opt, device, train_cfg.grad_clip)
        val_loss = validate_gnn(model, val_loader, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model":model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch":epoch,
                "val_loss": val_loss,
                "train_cfg": train_cfg.to_dict()
            }, path)

        if verbose:
            dt = time.perf_counter() - t0
            print(f"epoch {epoch:3d} train {train_loss:.5f} val {val_loss:.5f} "
                  f"({dt:.1f}s) {'*' if val_loss == best_val else ''}")

    return history, path