import numpy as np

from afrans.config import GridConfig, TrainConfig


def test_gridconfig_shape_and_tag_are_y_first():
    """shape/tag use (grid_res_y, grid_res_x) -- row-major, y first. Mixing
    this up with the CLI's natural (NX, NY) order caused real bugs earlier
    in this project."""
    cfg = GridConfig(grid_res_x=256, grid_res_y=128)
    assert cfg.shape == (128, 256)
    assert cfg.tag == "128x256"


def test_gridconfig_from_mapping_roundtrip():
    """from_mapping must undo what np.savez does to scalars (0-d arrays)
    and must return ints as real ints, not floats -- the int/float coercion
    relies on f.type is int, which breaks if `from __future__ import
    annotations` is ever added to this file."""
    cfg = GridConfig(grid_res_x=64, grid_res_y=32, x_min=-1.0, x_max=2.0)
    # simulate what np.load() hands back: 0-d arrays, not plain python scalars
    mapping = {k: np.array(v) for k, v in cfg.to_dict().items()}

    rebuilt = GridConfig.from_mapping(mapping)

    assert rebuilt == cfg
    assert isinstance(rebuilt.grid_res_x, int)
    assert isinstance(rebuilt.x_min, float)


def test_gridconfig_cache_path_depends_on_resolution():
    """Two different resolutions must not collide on the same cache file."""
    a = GridConfig(grid_res_x=64, grid_res_y=32)
    b = GridConfig(grid_res_x=128, grid_res_y=64)
    assert a.cache_path("train") != b.cache_path("train")


def test_trainconfig_n_modes_coerced_to_tuple():
    """n_modes must end up a tuple regardless of what it's constructed
    with, so two configs built from a list vs a tuple compare equal and
    both are hashable (needed for checkpoint-identity checks)."""
    from_list = TrainConfig(n_modes=[32, 64])
    from_tuple = TrainConfig(n_modes=(32, 64))

    assert from_list.n_modes == (32, 64)
    assert isinstance(from_list.n_modes, tuple)
    assert from_list == from_tuple
    assert hash(from_list) == hash(from_tuple)


def test_run_name_fno_branch_encodes_resolution_and_weighting():
    """Two FNO runs that differ only in grid resolution or loss_weighting
    must not produce the same checkpoint filename -- this exact collision
    silently overwrote a real checkpoint earlier in this project."""
    grid_a = GridConfig(grid_res_x=64, grid_res_y=32)
    grid_b = GridConfig(grid_res_x=128, grid_res_y=64)

    plain = TrainConfig(model="fno", loss_weighting="none")
    weighted = TrainConfig(model="fno", loss_weighting="density")

    assert plain.run_name(grid_a) != plain.run_name(grid_b)
    assert plain.run_name(grid_a) != weighted.run_name(grid_a)


def test_run_name_gnn_branch_needs_no_grid_config():
    """The GNN never touches a grid -- run_name() must work with no
    grid_cfg argument at all, and the name must reflect GNN-specific
    fields (k_neighbors, n_layers), not FNO-only ones (n_modes)."""
    cfg = TrainConfig(model="gnn", k_neighbors=8, n_layers=4)
    name = cfg.run_name()
    assert "gnn" in name
    assert "k8" in name
    assert "l4" in name
