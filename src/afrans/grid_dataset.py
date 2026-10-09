"""
Grid dataset:
AirfRANS point clouds pre-interpolated onto a regular grid.
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial import Delaunay
from afrans.cache import load_grid_cache
from afrans.interpolate import point_to_grid
from afrans.config import GridConfig, FEATURES, POS, TARGETS



class AirfRANSGridDataset(Dataset):

    def __init__(self, x_grids, y_grids, pos=None, y_points_raw=None):
        self.x_grids = x_grids
        self.y_grids = y_grids
        self.pos = pos
        self.y_points_raw = y_points_raw

    @classmethod
    def from_dataset(cls, dataset, indices, normaliser, 
                     grid_cfg=None, keep_points=True, verbose=True):
        cfg = GridConfig() if grid_cfg is None else grid_cfg
        gx, gy = cfg.mesh()
        if verbose:
            print(f"Pre-interpolating {len(indices)} sims onto {cfg.tag} ...")

        x_grids, y_grids, pos, y_pts = [], [], [], []
        for i in indices:
            sim = dataset[i]
            p = sim[:, POS]
            tri = Delaunay(p)
            x_norm = normaliser.transform_feature(sim[:, FEATURES])
            y_norm = normaliser.transform_target(sim[:, TARGETS])
            x_grids.append(point_to_grid(x_norm, tri, gx, gy))
            y_grids.append(point_to_grid(y_norm, tri, gx, gy))
            if keep_points:
                pos.append(torch.from_numpy(p.astype(np.float32)))
                y_pts.append(torch.from_numpy(sim[:, TARGETS].astype(np.float32)))

        return cls(
            torch.from_numpy(np.stack(x_grids)),
            torch.from_numpy(np.stack(y_grids)),
            pos = pos or None,
            y_points_raw = y_pts or None
        )

    @classmethod
    def from_cache(cls, path, normaliser, grid_cfg=None):
        """Load a cache -- raw grids -- and normalise in one pass."""
        cache = load_grid_cache(path, grid_cfg)
        x = cache["x_grids"].astype(np.float32, copy=True)
        y = cache["y_grids"].astype(np.float32, copy=True)

        fmean = normaliser.feature_mean.astype(np.float32).reshape(1, -1, 1, 1)
        fstd = normaliser.feature_std.astype(np.float32).reshape(1, -1, 1, 1)
        tmean = normaliser.target_mean.astype(np.float32).reshape(1, -1, 1, 1)
        tstd = normaliser.target_std.astype(np.float32).reshape(1, -1, 1, 1)

        x -= fmean; x /= fstd
        y -= tmean; y /= tstd

        return cls(torch.from_numpy(x), torch.from_numpy(y))


    def __len__(self):
        return len(self.x_grids)

    def __getitem__(self, i):
        return self.x_grids[i], self.y_grids[i]