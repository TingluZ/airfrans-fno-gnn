from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from afrans.config import FEATURES, N_FEATURES, N_TARGETS, POS, TARGETS, GridConfig
from afrans.interpolate import point_to_grid


def build_grid_cache(dataset, indices, path, grid_cfg=None,
                     overwrite=False, verbose=True):
    """
    Interpolate indices onto the grid and write .npz, return the path
    """
    cfg=GridConfig() if grid_cfg is None else grid_cfg

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists -- pass overwrite=True to rebuild")
    path.parent.mkdir(parents=True, exist_ok=True)

    gx, gy = cfg.mesh()
    height, width = cfg.shape
    n = len(indices)

    x_grids = np.empty((n, N_FEATURES, height, width), dtype=np.float32)
    y_grids = np.empty((n, N_TARGETS, height, width), dtype=np.float32)

    for k, i in enumerate(indices):
        sim = dataset[i]
        tri = Delaunay(sim[:, POS])
        x_grids[k] = point_to_grid(sim[:, FEATURES], tri, gx, gy, fill_value=np.nan)
        y_grids[k] = point_to_grid(sim[:, TARGETS], tri, gx, gy, fill_value=np.nan)
        if verbose and (k+1)%25 == 0:
            print(f"{k+1}/{n}")

    # NaN marks grid cells outside the mesh's convex hull

    inside = ~np.isnan(y_grids[:, 0])
    np.nan_to_num(x_grids, copy=False, nan=0.0)
    np.nan_to_num(y_grids, copy=False, nan=0.0)

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, 
                 x_grids=x_grids, 
                 y_grids=y_grids,
                 inside=inside,
                 indices=np.asarray(indices, dtype=np.int64),
                 **cfg.to_dict())
    tmp.replace(path)

    if verbose:
        print(f"wrote {path} ({path.stat().st_size/1e6:.0f} MB)")
    
    return path

def load_grid_cache(path, grid_cfg=None):
    with np.load(path) as z:
        cache_cfg = GridConfig.from_mapping(z)
        if grid_cfg is not None and cache_cfg != grid_cfg:
            raise ValueError(
                f"cache at {path} was built with {cache_cfg}, "
                f"{grid_cfg} was requested"
            )
        return{
            "x_grids": z["x_grids"], # (n, 7, H, W)
            "y_grids": z["y_grids"], # (n, 4, H, W)
            "inside": z["inside"], # (n, H, W)
            "indices": z["indices"], # (n, )
            "cfg": cache_cfg,
        }