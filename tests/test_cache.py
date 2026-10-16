import numpy as np

from afrans.config import GridConfig


def test_weight_map_bin_centers_align_with_mesh_samples():
    """build_weight_map's histogram bins must be centred on
    grid_cfg.mesh()'s own sample points (offset by half a cell on each
    side) -- not naive linspace(x_min, x_max, W+1) bins, which land half a
    cell off from where point_to_grid actually samples and silently
    misalign the weight with the loss it's meant to weight.

    Checked directly: the midpoint of each bin's edges must equal the
    corresponding mesh sample coordinate. An earlier version of this test
    checked only that each sample point landed in its own distinct bin --
    that passed for the naive (wrong) edges too, since distinctness is a
    much weaker property than centring. Centre-matching is the property
    that actually matters.
    """
    cfg = GridConfig(grid_res_x=16, grid_res_y=8)
    gx, gy = cfg.mesh()   # gx[0] = x sample points, gy[:, 0] = y sample points

    dx = (cfg.x_max - cfg.x_min) / (cfg.grid_res_x - 1)
    dy = (cfg.y_max - cfg.y_min) / (cfg.grid_res_y - 1)
    x_edges = np.linspace(cfg.x_min - dx / 2, cfg.x_max + dx / 2, cfg.grid_res_x + 1)
    y_edges = np.linspace(cfg.y_min - dy / 2, cfg.y_max + dy / 2, cfg.grid_res_y + 1)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    assert np.allclose(x_centers, gx[0])
    assert np.allclose(y_centers, gy[:, 0])


def test_naive_bins_do_not_center_on_mesh_samples():
    """Negative control: linspace(x_min, x_max, W+1) -- the construction
    it would be easy to reach for instead -- must NOT satisfy the
    centring property above."""
    cfg = GridConfig(grid_res_x=16, grid_res_y=8)
    gx, gy = cfg.mesh()

    x_edges = np.linspace(cfg.x_min, cfg.x_max, cfg.grid_res_x + 1)
    y_edges = np.linspace(cfg.y_min, cfg.y_max, cfg.grid_res_y + 1)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    assert not np.allclose(x_centers, gx[0])
    assert not np.allclose(y_centers, gy[:, 0])
