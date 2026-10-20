"""
Export test-set predictions for the web explorer on the portfolio site

Runs the final FNO (density-weighted) and GNN checkpoints on all 200 held-out
test simulations -- the same models and normaliser as run_testset.ipynb -- and
writes a compact, static dataset a browser can load one case at a time:

    <out>/index.json          case list, per-case errors, binary layout, attribution
    <out>/cases/<i>.bin       one simulation: positions + truth/FNO/GNN fields

Each case is cropped to a window around the airfoil and downsampled: every
surface point is kept, the rest are sampled half in proportion to mesh density
(so the boundary layer stays resolved) and half in proportion to local cell
area (so the far field isn't empty).

    python scripts/export_web.py --out ../tinglu-portfolio/public/data/airfrans
    python scripts/export_web.py --out /tmp/airfrans-web --limit 3     # smoke test
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

from afrans.config import (CACHE_DIR, CKPT_DIR, POS, REPO_ROOT, SURFACE,
                           TARGET_NAMES, TARGETS, GridConfig, TrainConfig)
from afrans.graph_dataset import build_graph
from afrans.metrics import relative_l2
from afrans.models.fno import build_fno
from afrans.models.gnn import build_gnn
from afrans.normalise import Normaliser
from afrans.splits import simulation_names
from afrans.web_export import ATTRIBUTION, BOX, SOURCES, encode_case, sample_points

sys.path.append(str(REPO_ROOT / "scripts"))
from evaluate import load_full_sim, predict_sim  # noqa: E402

NOTICE = """# Data notice

This folder is a derivative of the AirfRANS dataset (Bonnet et al., NeurIPS 2022
Datasets and Benchmarks Track): https://github.com/Extrality/AirfRANS

Fields from the 200 `full` test simulations are cropped to a window around the
airfoil, downsampled, and stored at reduced precision alongside FNO and GNN
predictions made with https://github.com/TingluZ/airfrans-fno-gnn.

Like the original, this data is licensed under the Open Database License
(ODbL) v1.0: https://opendatacommons.org/licenses/odbl/1-0/

If you use it, please cite the original dataset:

```bibtex
@inproceedings{bonnet2022airfrans,
  title={AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions},
  author={Florent Bonnet and Jocelyn Ahmed Mazari and Paola Cinnella and Patrick Gallinari},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://arxiv.org/abs/2212.07564}
}
```
"""


def load_models():
    fno_ck = torch.load(CKPT_DIR / (TrainConfig(loss_weighting="density").run_name(GridConfig()) + ".pt"),
                        map_location="cpu", weights_only=False)
    fno_cfg = TrainConfig(**fno_ck["train_cfg"])
    fno = build_fno(fno_cfg)
    fno.load_state_dict(fno_ck["model"])
    fno.eval()

    gnn_ck = torch.load(CKPT_DIR / (TrainConfig(model="gnn").run_name() + ".pt"),
                        map_location="cpu", weights_only=False)
    gnn_cfg = TrainConfig(**gnn_ck["train_cfg"])
    gnn = build_gnn(gnn_cfg)
    gnn.load_state_dict(gnn_ck["model"])
    gnn.eval()
    return fno, GridConfig(**fno_ck["grid_cfg"]), gnn, gnn_cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output folder, e.g. ../tinglu-portfolio/public/data/airfrans")
    p.add_argument("--points", type=int, default=6000, help="points kept per case (surface included)")
    p.add_argument("--limit", type=int, help="export only the first N test sims")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)

    norm = Normaliser()
    norm.load(CACHE_DIR / "normaliser.npz")
    scale = norm.target_std.astype(np.float64)
    fno, fno_grid, gnn, gnn_cfg = load_models()

    names = simulation_names("full", train=False)
    if args.limit:
        names = names[:args.limit]

    cases, t0, total_bytes = [], time.time(), 0
    for i, name in enumerate(names):
        sim = load_full_sim(name)
        y_true = sim[:, TARGETS]
        surf = sim[:, SURFACE] > 0.5
        pos = sim[:, POS]

        y_fno = predict_sim(fno, norm, fno_grid, torch.device("cpu"), sim)
        data = build_graph(name, norm, k=gnn_cfg.k_neighbors)
        with torch.no_grad():
            y_gnn = norm.inverse_target(gnn(data.x, data.edge_index).numpy())

        # same metrics as run_testset.ipynb, on the full-resolution fields
        ref = norm.target_std * np.sqrt(surf.sum())
        err = {
            m: {"all": relative_l2(y, y_true).tolist(),
                "surface": relative_l2(y, y_true, mask=surf, ref_norm=ref).tolist()}
            for m, y in (("fno", y_fno), ("gnn", y_gnn))
        }

        idx = sample_points(pos, surf, args.points, np.random.default_rng(args.seed + i))
        n_surf = int(surf[idx].sum())  # surface points come first in idx
        blob = encode_case(pos[idx], {"truth": y_true[idx], "fno": y_fno[idx], "gnn": y_gnn[idx]}, scale)
        (out / "cases" / f"{i}.bin").write_bytes(blob)
        total_bytes += len(blob)

        u_in = sim[0, 2:4]
        cases.append({
            "id": i,
            "name": name,
            "aoa_deg": round(float(np.degrees(np.arctan2(u_in[1], u_in[0]))), 3),
            "u_inf": round(float(np.linalg.norm(u_in)), 3),
            "n_points": int(len(idx)),
            "n_surface": n_surf,
            "error": err,
        })
        if (i + 1) % 10 == 0 or i + 1 == len(names):
            print(f"    {i + 1}/{len(names)}  ({time.time() - t0:.0f}s)")

    index = {
        "version": 1,
        "created": date.today().isoformat(),
        "split": "full_test",
        "box": list(BOX),
        "channels": list(TARGET_NAMES),
        "units": {"vx": "m/s", "vy": "m/s", "pressure": "m^2/s^2 (kinematic)", "nu_t": "m^2/s"},
        "scale": scale.tolist(),
        "sources": SOURCES,
        "layout": "uint16 pos[n][2] (quantised over box), then float16 field[source][channel][n] "
                  "times scale[channel]; the first n_surface points lie on the airfoil",
        "models": {
            "fno": TrainConfig(loss_weighting="density").run_name(fno_grid),
            "gnn": TrainConfig(model="gnn").run_name(),
        },
        "attribution": ATTRIBUTION,
        "cases": cases,
    }
    (out / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    (out / "NOTICE.md").write_text(NOTICE)
    print(f"\nwrote {len(cases)} cases, {total_bytes / 1e6:.1f} MB, to {out}")


if __name__ == "__main__":
    main()
