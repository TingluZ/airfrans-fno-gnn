"""
Package the trained k-NN GNN and a spread of test airfoils for the API

Run where the checkpoint, normaliser and AirfRANS dataset live (your Mac):

    python scripts/export_serving.py --out serving_artifacts
    python scripts/export_serving.py --out serving_artifacts --upload tingluzhang/airfrans-gnn

--upload pushes the folder to a Hugging Face model repo with your local
login (`hf auth login`), creating the repo if needed. The Space downloads it
at startup. Case ids match the portfolio explorer's (position in the test set).

For each exported case it also re-runs the model through the serving path
(float32 arrays, afrans.inference) and prints the relative L2 error next to
the evaluation path's, so you can see the two agree before uploading.
"""

import argparse
import json
import time
from pathlib import Path

import airfrans as af
import numpy as np
import torch

from afrans.config import CACHE_DIR, CKPT_DIR, DATA_ROOT, TrainConfig
from afrans.graph_dataset import build_graph
from afrans.inference import load_gnn, predict_case
from afrans.metrics import relative_l2
from afrans.serving import save_case, save_model
from afrans.splits import simulation_names
from afrans.web_export import ATTRIBUTION

CARD = """---
license: odbl
tags: [graph-neural-network, cfd, airfrans, surrogate-model]
---

# AirfRANS GraphSAGE surrogate (serving artifacts)

A 4-layer GraphSAGE network (PyTorch Geometric `SAGEConv`, 32 hidden channels,
~5k parameters) on an 8-nearest-neighbour graph, predicting velocity, kinematic
pressure and turbulent viscosity around 2D airfoils. Code, training and
evaluation: https://github.com/TingluZ/airfrans-fno-gnn

Contents: `model.pt` (weights + config), `normaliser.npz`, and {n} test airfoils
from AirfRANS under `cases/` (mesh positions, wall distance, normals, surface
mask and the CFD solution), listed in `cases.json`.

## Data notice

The cases are a derivative of the AirfRANS dataset (Bonnet et al., NeurIPS 2022
Datasets and Benchmarks Track, {paper_url}), licensed under the Open Database
License (ODbL) v1.0, and shared under the same licence. Source: {data_url}
"""


def pick_cases(names, n):
    """n test cases spread evenly over angle of attack (AirfRANS names carry
    u_inf and the angle in degrees as their 3rd and 4th fields)"""
    aoa = np.array([float(name.split("_")[3]) for name in names])
    order = np.argsort(aoa)
    return sorted(order[np.linspace(0, len(order) - 1, n).round().astype(int)].tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--n-cases", type=int, default=20)
    p.add_argument("--checkpoint", default=str(CKPT_DIR / (TrainConfig(model="gnn").run_name() + ".pt")))
    p.add_argument("--upload", metavar="REPO_ID", help="Hugging Face model repo, e.g. tingluzhang/airfrans-gnn")
    args = p.parse_args()

    out = Path(args.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)

    model, cfg, norm = load_gnn(args.checkpoint, CACHE_DIR / "normaliser.npz")
    save_model(out / "model.pt", model, cfg)
    norm.save(out / "normaliser.npz")

    names = simulation_names("full", train=False)
    cases, t0 = [], time.time()
    for i in pick_cases(names, args.n_cases):
        s = af.Simulation(root=str(DATA_ROOT), name=names[i])
        pos, sdf, normals = s.position, s.sdf, s.normals
        aoa, u_inf = float(s.angle_of_attack), float(s.inlet_velocity)
        truth = np.concatenate([s.velocity, s.pressure, s.nu_t], axis=-1)
        save_case(out / "cases" / f"{i}.npz", pos, sdf, normals, s.surface, truth)

        # evaluation path, exactly as evaluate_gnn.py / export_web.py run it
        data = build_graph(names[i], norm, k=cfg.k_neighbors)
        with torch.no_grad():
            y_eval = norm.inverse_target(model(data.x, data.edge_index).numpy())

        # serve-path check: the arrays exactly as the API will load them
        y_serve = predict_case(model, cfg, norm, pos.astype(np.float32), sdf.astype(np.float32),
                               normals.astype(np.float32), aoa, u_inf)
        e_eval, e_serve = relative_l2(y_eval, truth), relative_l2(y_serve, truth)
        print(f"  case {i:3d}  aoa {np.degrees(aoa):6.2f}  rel L2 eval {np.round(e_eval, 4)}  "
              f"serve {np.round(e_serve, 4)}  ({time.time() - t0:.0f}s)")

        cases.append({"id": i, "name": names[i], "aoa_deg": round(float(np.degrees(aoa)), 3),
                      "u_inf": round(u_inf, 3), "n_points": int(len(pos)), "file": f"cases/{i}.npz"})

    (out / "cases.json").write_text(json.dumps(cases, indent=2))
    (out / "README.md").write_text(CARD.format(n=len(cases), **ATTRIBUTION))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nwrote {len(cases)} cases, {size / 1e6:.0f} MB, to {out}")

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.upload, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=args.upload, repo_type="model", folder_path=str(out),
                          commit_message="Export serving artifacts")
        print(f"uploaded to https://huggingface.co/{args.upload}")


if __name__ == "__main__":
    main()
