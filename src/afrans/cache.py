from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

import airfrans as af

from afrans.config import CACHE_DIR, DATA_ROOT, FEATURES, N_FEATURES, N_TARGETS, POS, TARGETS, GridConfig
from afrans.interpolate import point_to_grid

def weight_map_path(grid_cfg):
    return CACHE_DIR / f"weight_map_{grid_cfg.tag}.npz"

def build_weight_map(names, grid_cfg, log_scale=True, clip_percentile=99):
    """
    Density of mesh points per grid cell averaged over 'names' simulations

    Bin edges are centred on grid_cfg.mesh()'s own sample points
    (offset by half a cell on each side), not naive linspace(x_min, x_max, W+1) bins,
    which would land half a cell off from where point_to_gridactually samples, 
    and that will misalign the weight with the loss it's meant to weight
    
    clip_percentile caps the tail after log1p:
    the airfoil surface is a thin lin crammed with mesh points, so some cells can reach
    10-15x the mean even after log-compression
    """
    H, W = grid_cfg.shape
    dx = (grid_cfg.x_max - grid_cfg.x_min) / (grid_cfg.grid_res_x - 1)
    dy = (grid_cfg.y_max - grid_cfg.y_min) / (grid_cfg.grid_res_y - 1)
    x_edges = np.linspace(grid_cfg.x_min - dx/2, grid_cfg.x_max + dx/2, W + 1)
    y_edges = np.linspace(grid_cfg.y_min - dy/2, grid_cfg.y_max + dy/2, H + 1)

    counts = np.zeros((H, W), dtype=np.float64)
    for name in names:
        s = af.Simulation(root=str(DATA_ROOT), name=name)
        hist, _, _ = np.histogram2d(s.position[:, 1], s.position[:, 0],
                                    bins = [y_edges, x_edges])
        counts += hist
    counts /= len(names)

    # Raw counts span orders of magnitude
    # near-zero far field, very high at the surface
    # log1p compresses the range, normalising to mean 1 keeps the overall
    # loss scale close to unweighted MSE
    weight = np.log1p(counts) if log_scale else counts
    if clip_percentile is not None:
        weight = np.clip(weight, None, np.percentile(weight, clip_percentile))
    weight /= weight.mean()

    return weight.astype(np.float32)

def save_weight_map(weight, grid_cfg):
    path = weight_map_path(grid_cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, weight=weight, **grid_cfg.to_dict())
    return path

def load_weight_map(grid_cfg):
    path = weight_map_path(grid_cfg)
    with np.load(path) as z:
        cached_cfg = GridConfig.from_mapping(z)
        if cached_cfg != grid_cfg:
            raise ValueError(f"weight map at {path} was built for {cached_cfg}, "
                             f"{grid_cfg} requested")
        return z["weight"]

def get_weight_map(names, grid_cfg, log_scale=True, clip_percentile=99, overwrite=False):
    """
    Load the cached weight map, building and saving it
    overwrite=True forces a rebuild
    """
    if weight_map_path(grid_cfg).exists() and not overwrite:
        return load_weight_map(grid_cfg)

    weight = build_weight_map(names, grid_cfg, log_scale=log_scale,
                              clip_percentile=clip_percentile)
    save_weight_map(weight, grid_cfg)

    return weight

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