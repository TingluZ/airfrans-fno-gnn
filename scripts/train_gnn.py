import argparse
import json

import torch
from torch_geometric.loader import DataLoader

from afrans.config import CACHE_DIR, TrainConfig
from afrans.engine import fit_gnn, resolve_device
from afrans.graph_dataset import AirfRANSGraphDataset
from afrans.models.gnn import build_gnn
from afrans.normalise import Normaliser
from afrans.splits import load_split, simulation_names

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, help="use only the first N train/val sims, e.g. 20")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--hidden-channels", type=int)
    p.add_argument("--n-layers", type=int)
    p.add_argument("--graph", choices=["knn", "mesh"],
                   help="knn (default): k nearest neighbours; mesh: the real CFD mesh connectivity")
    p.add_argument("--k", type=int, help="k in the k-NN graph (ignored for --graph mesh)")
    p.add_argument("--num-workers", type=int,
                   help="worker processes to prefetch/build graphs in parallel with training")
    p.add_argument("--seed", type=int)
    p.add_argument("--no-slip-weight", type=float)
    p.add_argument("--tag", help="checkpoint filename suffix, e.g. 'ns_sweep_1'. "
                   "Defaults to 'smoke' whenever --limit is used, so a throwaway "
                   "run can never silently collide with a real checkpoint's filename.")
    args = p.parse_args()

    tag = args.tag if args.tag is not None else ("smoke" if args.limit else "")

    overrides = {
        "epochs":args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_channels":args.hidden_channels,
        "n_layers": args.n_layers,
        "k_neighbors": args.k,
        "graph": args.graph,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "no_slip_weight": args.no_slip_weight
    }

    train_cfg = TrainConfig(**{k: v for k, v in overrides.items() if v is not None},
                            model="gnn", device="cpu", tag=tag)
    torch.manual_seed(train_cfg.seed)

    norm =Normaliser()
    norm.load(CACHE_DIR / "normaliser.npz")

    split = load_split(names = simulation_names("full"))
    train_names, val_names = split["train_names"], split["val_names"]
    if args.limit:
        train_names = train_names[:args.limit]
        val_names = val_names[:max(1, args.limit // 9)]

    train_ds = AirfRANSGraphDataset(train_names, norm, k=train_cfg.k_neighbors, graph=train_cfg.graph)
    val_ds = AirfRANSGraphDataset(val_names, norm, k=train_cfg.k_neighbors, graph=train_cfg.graph)
    print(f"train {len(train_ds)} sims, val {len(val_ds)} sims, graph={train_cfg.graph}")

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers = train_cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size = train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)

    model = build_gnn(train_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{train_cfg.run_name()} device={resolve_device(train_cfg)} {n_params / 1e6:.1f}M params")

    history, ckpt_path = fit_gnn(model, train_loader, val_loader, train_cfg)

    history_path = ckpt_path.with_suffix(".json")
    history_path.write_text(json.dumps(history, indent=2))
    print(f"best checkpoint: {ckpt_path}")
    print(f"history:    {history_path}")


if __name__ == "__main__":
    main()
