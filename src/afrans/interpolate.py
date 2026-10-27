import numpy as np
from afrans.config import GridConfig
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay 


def point_to_grid(values, tri, gx, gy, fill_value=0.0):
    """values: (N, C) at the points triangulated by tri → grid: (C, H, W) at (gx, gy)."""

    
    interp = LinearNDInterpolator(tri, values, fill_value=fill_value)
    result = interp(gx, gy)

    return np.moveaxis(result, 2, 0).astype(np.float32)


def grid_to_point(grid, query_pos, grid_cfg, fill_value=None):

    y_axis, x_axis = grid_cfg.axes()

    values = np.moveaxis(grid, 0, -1)

    interp = RegularGridInterpolator(
        (y_axis, x_axis),
        values,
        method = "linear",
        bounds_error = False,
        fill_value = fill_value, # None -> linear extrapolation outside the box
    )

    return interp(query_pos[:, ::-1]).astype(np.float32)