import numpy as np
from scipy.spatial import Delaunay

from afrans.config import GridConfig
from afrans.interpolate import point_to_grid


def test_point_to_grid_recovers_query_coordinates():
    """Feed point_to_grid the points' own (x, y) as its 'values' and it
    must reproduce grid_cfg.mesh()'s exact query locations everywhere
    inside the convex hull. This is the manual sanity check that caught
    transposed-axis/moveaxis bugs during development, now automated so it
    can't silently regress."""
    cfg = GridConfig(grid_res_x=20, grid_res_y=10)

    rng = np.random.default_rng(0)
    pos = rng.uniform(
        low=[cfg.x_min, cfg.y_min], high=[cfg.x_max, cfg.y_max], size=(500, 2)
    )
    # guarantee the convex hull actually covers the whole grid box
    corners = np.array([[cfg.x_min, cfg.y_min], [cfg.x_max, cfg.y_min],
                        [cfg.x_min, cfg.y_max], [cfg.x_max, cfg.y_max]])
    pos = np.vstack([pos, corners])

    tri = Delaunay(pos)
    gx, gy = cfg.mesh()
    grid = point_to_grid(pos, tri, gx, gy)

    assert grid.shape == (2, *cfg.shape)
    assert np.allclose(grid[0], gx, atol=1e-4)
    assert np.allclose(grid[1], gy, atol=1e-4)


def test_point_to_grid_fill_value_used_outside_hull():
    """Query points entirely outside the data's convex hull must come
    back as fill_value, not extrapolated or left undefined."""
    # a tiny triangle in the corner of the domain -- most of the grid
    # box lies outside its convex hull
    cfg = GridConfig(grid_res_x=5, grid_res_y=5,
                     x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
    pos = np.array([[0.0, 0.0], [0.05, 0.0], [0.0, 0.05]])
    values = np.array([[1.0], [1.0], [1.0]])
    tri = Delaunay(pos)
    gx, gy = cfg.mesh()

    grid = point_to_grid(values, tri, gx, gy, fill_value=-1.0)

    assert grid[0, -1, -1] == -1.0  # far corner, well outside the tiny triangle
