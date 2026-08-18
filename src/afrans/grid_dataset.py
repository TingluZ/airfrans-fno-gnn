import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial import Delaunay
from afrans.interpolate import point_to_grid
from afrans.config import GridConfig, FEATURES, POS, TARGETS



class AirfRANSGridDataset(Dataset):
    def __init__(
            self, 
            dataset, 
            indices, 
            normaliser,
            grid_cfg = None,
            keep_points = True,
            verbose = True):

        self.cfg = GridConfig() if grid_cfg is None else grid_cfg
        self.normaliser = normaliser
        self.keep_points = keep_points
    
        gx, gy = self.cfg.mesh()
        if verbose:
            print(f"Pre-interpolating {len(indices)} sim on to {self.cfg.tag}...")

        self.samples = [self._build(dataset[i], gx, gy) for i in indices]

    def _build(self, sim, gx, gy):

        pos = sim[:POS]
        tri = Delaunay(pos)

        x_norm = self.normaliser.transform_feature(sim[:, FEATURES])
        y_norm = self.normaliser.transform_target(sim[:, TARGETS])
        
        item = {
            "x_grid": torch.from_numpy(point_to_grid(x_norm, tri, gx, gy)),
            "y_grid": torch.from_numpy(point_to_grid(y_norm, tri, gx, gy))
        }

        if self.keep_points:
            item["pos"] = torch.from_numpy(pos.astype(np.float32))
            item["y_points_raw"] = torch.from_numpy(sim[:, TARGETS].astype(np.float32))

        return item

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]


def collate(batch):
    x = torch.stack([b['x_grid'] for b in batch])
    y = torch.stack([b['y_grid'] for b in batch])
    return x, y, batch