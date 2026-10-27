"""
Point-space evaluation of a trained checkpoint against the val split

Reads val simulations directly with af.Simulation rather than the grid cache:
the cache holds no point data ( see cache.py) and 80 sims at ~0.2s each is ~16s in total

    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/fno_..._64x128.pt --suffix _smoke
"""

import argparse
import json

import airfrans as af
import numpy as np
import torch
from scipy.spatial import Delaunay

from afrans.config import (CACHE_DIR, CKPT_DIR, DATA_ROOT, FEATURES, POS,
                           REPO_ROOT, SURFACE, TARGET_NAMES, TARGETS, 
                           GridConfig, TrainConfig)
from afrans.engine import resolve_device
from afrans.interpolate import grid_to_point, point_to_grid
from afrans.metrics import relative_l2, summarize
from afrans.models.fno import build_fno
from afrans.normalise import Normaliser
from afrans.splits import load_split, simulation_names

def load_full_sim(name):
    """
    Ressemble AirfRANS's own 12-column layout from a new Simulation --
    must match af.dataset.load's column order exactly
    """

    s = af.Simulation(root=str(DATA_ROOT), name=name)
    inlet = (np.array([np.cos(s.angle_of_attack), np.sin(s.angle_of_attack)])
             * s.inlet_velocity).reshape(1, 2) * np.ones_like(s.sdf) #signed distance field
    return np.concatenate([s.position,
                           inlet,
                           s.sdf,
                           s.normals,
                           s.velocity,
                           s.pressure,
                           s.nu_t,
                           s.surface.reshape(-1, 1)],
                           axis=-1)

def predict_sim(model, normaliser, grid_cfg, device, sim):
    """
    One simulation, point cloud -> grid -> model -> point cloud
    """
    pos = sim[:, POS]
    tri = Delaunay(pos)
    gx, gy = grid_cfg.mesh()

    x_norm = normaliser.transform_feature(sim[:, FEATURES])
    x_grid = point_to_grid(x_norm, tri, gx, gy)
    x = torch.from_numpy(x_grid).unsqueeze(0).to(device)

    with torch.no_grad():
        y_grid = model(x)[0].cpu().numpy()

    y_pred_norm = grid_to_point(y_grid, pos, grid_cfg)
    return normaliser.inverse_target(y_pred_norm)



def main():
    p = argparse.ArgumentParser(description=__doc__)
    # train.py always trains with the density-weighted loss
    default_ckpt = CKPT_DIR / (TrainConfig(loss_weighting="density").run_name(GridConfig()) + ".pt")
    p.add_argument("--checkpoint", default=str(default_ckpt))
    p.add_argument("--suffix", default="", help="normaliser suffix, e.g. _smoke")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--limit", type=int, help="check only the first N sims, e.g. 20")
    args = p.parse_args()

    # weights_only=False: this checkpoint carries plain-dict config alongside
    # tensors, and it's a file this project wrote itself, not a download 
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_cfg = TrainConfig(**ck["train_cfg"])
    grid_cfg = GridConfig(**ck["grid_cfg"])
    device = resolve_device(train_cfg)

    model = build_fno(train_cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    norm = Normaliser()
    norm.load(CACHE_DIR / f"normaliser{args.suffix}.npz")

    split = load_split(names=simulation_names("full"))
    names = split[f"{args.split}_names"]
    if args.limit:
        names = names[:args.limit]
    print(f"{args.checkpoint} (epoch {ck['epoch']}, val_loss {ck['val_loss']:.5f})")
    print(f"evaluating on {len(names)} {args.split} simuations, grid {grid_cfg.tag}...")

    all_errs, surf_errs = [], []
    for i, name in enumerate(names):
        sim = load_full_sim(name)
        y_true = sim[:, TARGETS]
        surf = sim[:, SURFACE] > 0.5
        n_surf = surf.sum()

        y_pred = predict_sim(model, norm, grid_cfg, device, sim)
        all_errs.append(relative_l2(y_pred, y_true))
        surf_errs.append(relative_l2(
            y_pred, y_true, mask=surf,
            ref_norm=norm.target_std * np.sqrt(n_surf)
            ))

        if (i+1) % 20 == 0:
            print(f"    {i + 1}/{len(names)}")

    results = {
        "checkpoints": args.checkpoint,
        "split": args.split,
        "n_sims": len(names),
        "all_points": summarize(all_errs, TARGET_NAMES),
        "surface_points": summarize(surf_errs, TARGET_NAMES),
    }

    print(f"\n{'target':10s} {'all pts (mean±std)':22s} {'surface (rel. to global std)':22s}")
    for name in TARGET_NAMES:
        a, s = results["all_points"][name], results["surface_points"][name]
        print(f"{name:10s} {a['mean']:6.1%} ± {a['std']:5.1%}       "
              f"{s['mean']:6.1%} ± {s['std']:5.1%}")

    tag = "" if args.split == "val" else f"_{args.split}"
    out_path = REPO_ROOT / f"results{tag}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")

if __name__=="__main__":
    main()