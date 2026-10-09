"""
Train a single model against a pre-built grid cache

python scripts/train.py
python scripts/train.py --epochs 2 --grid-res 64 128 --suffix _smoke
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader


from afrans.grid_dataset import AirfRANSGridDataset
from afrans.config import CACHE_DIR, GridConfig, TrainConfig
from afrans.engine import fit, resolve_device
from afrans.models.fno import build_fno
from afrans.normalise import Normaliser
from afrans.cache import get_weight_map
from afrans.splits import load_split, simulation_names

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suffix", default="", help="cache/normaliser suffix, e.g. _smoke")
    p.add_argument("--grid-res", type=int, nargs=2, metavar=("NX", "NY"))
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--hidden-channels", type=int)
    p.add_argument("--n-modes", type=int, nargs=2, metavar=("MY", "MX"))
    p.add_argument("--seed", type=int)
    args = p.parse_args()

    grid_kwargs = {}
    if args.grid_res:
        grid_kwargs["grid_res_x"], grid_kwargs["grid_res_y"] = args.grid_res
    grid_cfg = GridConfig(**grid_kwargs)

    overrides = {
        "epochs":args.epochs,
        "batch_size": args.batch_size, 
        "lr": args.lr, 
        "hidden_channels": args.hidden_channels,
        "n_modes": tuple(args.n_modes) if args.n_modes else None,
        "seed": args.seed,
    }

    train_cfg = TrainConfig(**{k:v for k, v in overrides.items() if v is not None},
                            loss_weighting="density", grad_clip=1.0)
    torch.manual_seed(train_cfg.seed)

    norm = Normaliser()
    norm.load(CACHE_DIR / f"normaliser{args.suffix}.npz")

    train_ds = AirfRANSGridDataset.from_cache(
        grid_cfg.cache_path("train" + args.suffix), norm, grid_cfg
    )
    val_ds = AirfRANSGridDataset.from_cache(
        grid_cfg.cache_path("val" + args.suffix), norm, grid_cfg
    )
    print(f"train {len(train_ds)} sims, val {len(val_ds)} sims, grid {grid_cfg.tag}")

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=train_cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size,
                            shuffle=False, num_workers=train_cfg.num_workers)

    model = build_fno(train_cfg)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{train_cfg.run_name(grid_cfg)} "
          f"device={resolve_device(train_cfg)} {n_params / 1e6: .1f}M params")

    split = load_split(names=simulation_names("full"))
    weight = get_weight_map(split["train_names"], grid_cfg)
    history, ckpt_path = fit(model, train_loader, val_loader, train_cfg, grid_cfg, weight)

    history_path = ckpt_path.with_suffix(".json")
    history_path.write_text(json.dumps(history, indent=2))
    print(f"best checkpoint: {ckpt_path}")
    print(f"history:        {history_path}")

if __name__ == "__main__":
    main()