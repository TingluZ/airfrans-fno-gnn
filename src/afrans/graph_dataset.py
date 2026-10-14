"""
Graph dataset: AirfRANS point clouds as k-NN graphs, PyG-native

Graphs are build lazily (one per __getitem__ call, every epoch)
rather than pre-built and stored: edege_index alone is  ~26 MB 
per simulation, and holding all 720 training praghs in memory
at once  would be ~19 GB. Rebuilding costs ~0.4s/sim/epoch instead

Normalisation reuses the same normaliser.npz the FNO trained with:
it was deliberately fit on points so both models share one fair
normalisation
"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import airfrans as af

from afrans.config import DATA_ROOT, FEATURES, POS, SURFACE, TARGETS

def build_graph(name, normaliser, k=8):
    """
    One simulation -> one PyG Data object, fully on normalisted space

    Bare connectively only - first benchmark version: edge_index and nothing else in the edges
    No edge_attr (relative position, distance) 
    """
    s = af.Simulation(root=str(DATA_ROOT), name=name)
    inlet = (np.array([np.cos(s.angle_of_attack), np.sin(s.angle_of_attack)]) \
             * s.inlet_velocity).reshape(1, 2) * np.ones_like(s.sdf)

    sim = np.concatenate([s.position, inlet, s.sdf, s.normals,
                          s.velocity, s.pressure, s.nu_t, 
                          s.surface.reshape(-1, 1)], axis=-1)

    pos = sim[:, POS]
    N = pos.shape[0]
    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=k+1)
    idx = idx[:, 1:] # drop self -- column 0 of every row is always the point itself

    src = np.repeat(np.arange(N), k) # repeat points' index k times (k nearest neighbors)
    dst = idx.ravel() # flatten the index
    edge_index = torch.from_numpy(np.stack([src, dst])).long()
    # k-NN is not symmetric
    # this scans the edge list the adds the missing reverse edge
    # wherever only one direction exists
    edge_index = to_undirected(edge_index) 

    x = normaliser.transform_feature(sim[:, FEATURES]).astype(np.float32)
    y = normaliser.transform_target(sim[:, TARGETS]).astype(np.float32)

    return Data(
        x = torch.from_numpy(x),
        edge_index = edge_index,
        y=torch.from_numpy(y),
        pos = torch.from_numpy(pos.astype(np.float32)),
        surface = torch.from_numpy(sim[:, SURFACE] >0.5)
    )

class AirfRANSGraphDataset(torch.utils.data.Dataset):
    def __init__(self, names, normaliser, k=8):
        self.names = names
        self.normaliser = normaliser
        self.k = k

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return build_graph(self.names[i], self.normaliser, self.k)