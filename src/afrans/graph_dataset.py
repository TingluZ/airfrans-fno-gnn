"""
Graph dataset: AirfRANS simulations as PyG graphs, one node per mesh point.

Two ways to connect the nodes (`graph=`):
  "knn"  -- k nearest neighbours in (x, y); a proxy for connectivity
  "mesh" -- the real CFD mesh: the four sides of every quadrilateral cell

Graphs are built lazily (one per __getitem__ call, every epoch) rather than
pre-built and stored: a k-NN edge_index alone is ~26 MB per simulation, and
holding all 720 training graphs in memory at once would be ~19 GB.
Rebuilding is cheap (~0.4 s for k-NN, ~0.04 s for the mesh graph).

Normalisation reuses the same normaliser.npz the FNO trained with: it was
deliberately fit on points, so both models share one fair normalisation.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from afrans.config import DATA_ROOT, FEATURES, POS, SURFACE, TARGETS

VTK_QUAD = 9


def edges_from_quads(quads, n_nodes):
    """(n_cells, 4) quad vertex ids -> (2, E) directed edge_index, both directions.

    VTK stores a quad's vertices in cyclic order, so its sides are consecutive
    pairs (0-1, 1-2, 2-3, 3-0) -- not the diagonals. Cells share sides, so
    duplicates are removed before both directions are added.
    """
    q = np.asarray(quads, dtype=np.int64)
    a = np.concatenate([q[:, 0], q[:, 1], q[:, 2], q[:, 3]])
    b = np.concatenate([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])
    # one integer per undirected edge: np.unique on a 1-D key is ~15x faster than on rows
    key = np.unique(np.minimum(a, b) * n_nodes + np.maximum(a, b))
    lo, hi = key // n_nodes, key % n_nodes
    return torch.from_numpy(np.stack([np.concatenate([lo, hi]),
                                      np.concatenate([hi, lo])])).long()


def mesh_edges(s):
    cells = s.internal.cells_dict
    if [int(c) for c in cells] != [VTK_QUAD]:
        raise ValueError(f"expected an all-quad mesh, got cell types {list(cells)}")
    return edges_from_quads(cells[VTK_QUAD], s.position.shape[0])


def knn_edges(pos, k):
    N = pos.shape[0]
    _, idx = cKDTree(pos).query(pos, k=k+1)
    idx = idx[:, 1:] # drop self -- column 0 of every row is always the point itself

    src = np.repeat(np.arange(N), k) # repeat points' index k times (k nearest neighbors)
    dst = idx.ravel() # flatten the index
    edge_index = torch.from_numpy(np.stack([src, dst])).long()
    # k-NN is not symmetric: this adds the missing reverse edge
    # wherever only one direction exists
    return to_undirected(edge_index)


def node_features(pos, sdf, normals, aoa, u_inf):
    """
    Raw (unnormalised) model inputs in COLS order, shape (N, 7).

    aoa is in radians, as AirfRANS stores it. AirfRANS sets the angle of
    attack through the inflow direction, not by rotating the airfoil, so the
    same mesh with a different (aoa, u_inf) is a valid input.
    """
    sdf = np.asarray(sdf).reshape(-1, 1)
    inlet = np.array([np.cos(aoa), np.sin(aoa)]) * u_inf * np.ones((len(sdf), 2))
    return np.concatenate([pos, inlet, sdf, normals], axis=-1)


def build_graph(name, normaliser, k=8, graph="knn"):
    """
    One simulation -> one PyG Data object, fully in normalised space.

    Bare connectivity only: edge_index and nothing else on the edges
    (no edge_attr such as relative position or distance).
    """
    import airfrans as af  # only needed with the raw dataset, not for serving

    s = af.Simulation(root=str(DATA_ROOT), name=name)
    features = node_features(s.position, s.sdf, s.normals,
                             s.angle_of_attack, s.inlet_velocity)

    sim = np.concatenate([features,
                          s.velocity, s.pressure, s.nu_t,
                          s.surface.reshape(-1, 1)], axis=-1)

    pos = sim[:, POS]
    if graph == "knn":
        edge_index = knn_edges(pos, k)
    elif graph == "mesh":
        edge_index = mesh_edges(s)
    else:
        raise ValueError(f"graph must be 'knn' or 'mesh', got {graph!r}")

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
    def __init__(self, names, normaliser, k=8, graph="knn"):
        self.names = names
        self.normaliser = normaliser
        self.k = k
        self.graph = graph

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return build_graph(self.names[i], self.normaliser, self.k, self.graph)
