"""
GNN inference on arrays: no raw dataset, no airfrans package

What the API in serve/ uses. Training and evaluation keep going through
graph_dataset.build_graph, which reads a Simulation and calls the same
node_features() and knn_edges(), so served and evaluated inputs match.
"""

import numpy as np
import torch

from afrans.config import TrainConfig
from afrans.graph_dataset import knn_edges, node_features
from afrans.models.gnn import build_gnn
from afrans.normalise import Normaliser


def load_gnn(ckpt_path, normaliser_path):
    """
    Checkpoint written by fit_gnn (or the slimmed copy export_serving.py
    writes) + its normaliser -> (model in eval mode, TrainConfig, Normaliser)
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ck["train_cfg"])
    model = build_gnn(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    norm = Normaliser()
    norm.load(normaliser_path)
    return model, cfg, norm


@torch.no_grad()
def predict(model, normaliser, features, edge_index):
    """
    Raw (N, 7) node features + (2, E) edge_index -> (N, 4) targets in
    physical units (vx, vy in m/s, kinematic pressure in m^2/s^2, nu_t in m^2/s)
    """
    x = normaliser.transform_feature(features).astype(np.float32)
    out = model(torch.from_numpy(x), edge_index)
    return normaliser.inverse_target(out.numpy())


def predict_case(model, cfg, normaliser, pos, sdf, normals, aoa, u_inf, edge_index=None):
    """
    One airfoil mesh at a chosen inflow. aoa in radians, u_inf in m/s.

    edge_index depends only on the mesh, not on the inflow, so a caller
    serving many requests for the same airfoil can pass a cached one.
    """
    if edge_index is None:
        if cfg.graph != "knn":
            raise ValueError(f"only k-NN models can be served from arrays, this one uses graph={cfg.graph!r}")
        edge_index = knn_edges(pos, cfg.k_neighbors)
    features = node_features(pos, sdf, normals, aoa, u_inf)
    return predict(model, normaliser, features, edge_index)
