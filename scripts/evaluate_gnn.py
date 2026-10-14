import argparse
import json

import numpy as np
import torch

from afrans.config import CACHE_DIR, CKPT_DIR, REPO_ROOT, TARGET_NAMES, TrainConfig
from afrans.engine import resolve_device
from afrans.graph_dataset import build_graph
from afrans.metrics import relative_l2, summarize
from afrans.models.gnn import build_gnn
from afrans.normalise import Normaliser
from afrans.splits import load_split, simulation_names

def main():
    p = argparse.ArgumentParser(description=__doc__)
    default_ckpt = CKPT_DIR / (TrainConfig(model="gnn").run_name() + ".pt")
    p.add_argument("--checkpoint", default=str(default_ckpt))
    p.add_argument("--limit", type=int, help="check only the first N val sims, e.g. 20")
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_cfg = TrainConfig(**ck["train_cfg"])
    device = resolve_device(train_cfg)

    model = build_gnn(train_cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    norm = Normaliser()
    norm.load(CACHE_DIR / "normaliser.npz")

    val_names = load_split(names=simulation_names("full"))["val_names"]
    if args.limit:
        val_names = val_names[:args.limit]
    print(f"{args.checkpoint} (epoch {ck['epoch']}, val_loss {ck['val_loss']:.5f})")
    print(f"evaluating on {len(val_names)} val simulations, k={train_cfg.k_neighbors}...")

    all_errs, surf_errs = [], []
    for i, name in enumerate(val_names):
        data = build_graph(name, norm, k=train_cfg.k_neighbors).to(device)

        with torch.no_grad():
            pred_norm = model(data.x, data.edge_index)

        pred = norm.inverse_target(pred_norm.cpu().numpy())
        true = norm.inverse_target(data.y.cpu().numpy())

        surf = data.surface.cpu().numpy()
        n_surf = surf.sum()

        all_errs.append(relative_l2(pred, true))
        surf_errs.append(relative_l2(
            pred, true, mask=surf,
            ref_norm=norm.target_std * np.sqrt(n_surf)
        ))

        if (i+1) % 20 == 0:
            print(f"    {i+1}/{len(val_names)}")

    results = {
        "checkpoint": args.checkpoint,
        "n_val_sims": len(val_names),
        "all_points": summarize(all_errs, TARGET_NAMES),
        "surface_points": summarize(surf_errs, TARGET_NAMES),
    }

    print(f"\n{'target':10s} {'all pts (mean±std)':22s} {'surface (rel. to global std)':22s}")
    for name in TARGET_NAMES:
        a, s = results["all_points"][name], results["surface_points"][name]
        print(f"{name:10s} {a['mean']:6.1%} ± {a['std']:5.1%}       "
              f"{s['mean']:6.1%} ± {s['std']:5.1%}")

    out_path = REPO_ROOT / "results_gnn.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")

if __name__ == "__main__":
    main()