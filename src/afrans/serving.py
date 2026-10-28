"""
On-disk layout of the artifacts the API serves, shared by
scripts/export_serving.py (writes them) and serve/app.py (reads them)

    <dir>/model.pt          GNN state_dict + TrainConfig (no optimizer state)
    <dir>/normaliser.npz    the training normaliser
    <dir>/cases.json        [{id, name, aoa_deg, u_inf, n_points, file}, ...]
    <dir>/cases/<id>.npz    pos (N, 2), sdf (N,), normals (N, 2), surface (N,), truth (N, 4)

Kept free of airfrans so the API image doesn't need it.
"""

import json
from pathlib import Path

import numpy as np
import torch

from afrans.config import TrainConfig
from afrans.models.gnn import build_gnn
from afrans.normalise import Normaliser

# AirfRANS sampling ranges (Bonnet et al., 2022): Reynolds 2e6 to 6e6 on a unit
# chord with nu = 1.56e-5 m^2/s, angle of attack -5 to 15 degrees. The model has
# never seen inflows outside this box, so the API refuses them.
NU = 1.56e-5
U_MIN, U_MAX = 2e6 * NU, 6e6 * NU      # ~31.2 to ~93.6 m/s
AOA_MIN_DEG, AOA_MAX_DEG = -5.0, 15.0


def save_model(path, model, train_cfg):
    torch.save({"model": model.state_dict(), "train_cfg": train_cfg.to_dict()}, path)


def save_case(path, pos, sdf, normals, surface, truth):
    """float32 halves the size; positions keep ~1e-7 chord precision near the wall"""
    np.savez_compressed(path,
                        pos=pos.astype(np.float32),
                        sdf=np.asarray(sdf, np.float32).reshape(-1),
                        normals=normals.astype(np.float32),
                        surface=np.asarray(surface, bool).reshape(-1),
                        truth=truth.astype(np.float32))


def load_case(path):
    with np.load(path) as z:
        return {k: z[k] for k in ("pos", "sdf", "normals", "surface", "truth")}


def load_manifest(root):
    return json.loads((Path(root) / "cases.json").read_text())


def make_fixture(root, n_cases=2, n_points=3000, seed=0):
    """
    A tiny, fully synthetic artifact folder: random-weight GNN, identity
    normaliser and blob-shaped "airfoils". For tests and the CI smoke test,
    where the real artifacts (and the dataset) aren't available.
    """
    root = Path(root)
    (root / "cases").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    cfg = TrainConfig(model="gnn")
    save_model(root / "model.pt", build_gnn(cfg), cfg)

    norm = Normaliser()
    norm.fit_from_stats(np.zeros(7), np.ones(7), np.zeros(4), np.ones(4), eps=0.0)
    norm.save(root / "normaliser.npz")

    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n_cases):
        n_surf = n_points // 10
        t = np.linspace(0, 2 * np.pi, n_surf, endpoint=False)
        surf_pos = np.stack([0.5 + 0.5 * np.cos(t), 0.06 * np.sin(t)], axis=1)
        far = rng.uniform([-0.6, -0.7], [2.2, 0.7], size=(n_points - n_surf, 2))
        pos = np.concatenate([surf_pos, far])
        surface = np.arange(n_points) < n_surf
        sdf = np.where(surface, 0.0, np.abs(pos[:, 1]) + 1e-3)
        normals = np.zeros((n_points, 2))
        normals[:n_surf] = np.stack([np.cos(t), np.sin(t)], axis=1)
        truth = rng.normal(size=(n_points, 4))
        save_case(root / "cases" / f"{i}.npz", pos, sdf, normals, surface, truth)
        cases.append({"id": i, "name": f"synthetic_{i}", "aoa_deg": 2.0 + i,
                      "u_inf": 50.0, "n_points": n_points, "file": f"cases/{i}.npz"})
    (root / "cases.json").write_text(json.dumps(cases, indent=2))
    return root
