import numpy as np

from afrans.web_export import BOX, SOURCES, encode_case, quantise_pos, sample_points, wall_layers


def _cloud(n=5000, n_surf=200, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform([-1, -1], [3, 1], size=(n, 2))
    surf = np.zeros(n, bool)
    surf[:n_surf] = True
    pos[:n_surf] = rng.uniform([0, -0.05], [1, 0.05], size=(n_surf, 2))
    return pos, surf


def test_sample_keeps_every_surface_point_first():
    pos, surf = _cloud()
    idx = sample_points(pos, surf, 1000, np.random.default_rng(0))
    n_surf = surf.sum()
    assert len(idx) == 1000
    assert surf[idx[:n_surf]].all() and not surf[idx[n_surf:]].any()
    assert len(np.unique(idx)) == len(idx)


def test_sample_stays_inside_box():
    pos, surf = _cloud()
    idx = sample_points(pos, surf, 1000, np.random.default_rng(0))
    x0, x1, y0, y1 = BOX
    p = pos[idx]
    assert (p[:, 0] >= x0).all() and (p[:, 0] <= x1).all()
    assert (p[:, 1] >= y0).all() and (p[:, 1] <= y1).all()


def test_sample_returns_all_candidates_when_budget_exceeds_them():
    pos, surf = _cloud(n=300, n_surf=50)
    idx = sample_points(pos, surf, 10_000, np.random.default_rng(0))
    x0, x1, y0, y1 = BOX
    inside = (pos[:, 0] >= x0) & (pos[:, 0] <= x1) & (pos[:, 1] >= y0) & (pos[:, 1] <= y1)
    assert len(idx) == inside.sum()


def test_quantise_round_trip_is_within_resolution():
    pos, _ = _cloud()
    x0, x1, y0, y1 = BOX
    pos = np.clip(pos, [x0, y0], [x1, y1])
    q = quantise_pos(pos).astype(float) / 65535
    back = np.array([x0, y0]) + q * np.array([x1 - x0, y1 - y0])
    assert np.abs(back - pos).max() < 3e-5


def test_encode_layout_and_scaling():
    n, c = 7, 4
    pos = np.full((n, 2), 0.5)
    scale = np.array([10.0, 10.0, 1000.0, 1e-4])
    fields = {s: np.tile(scale * (k + 1), (n, 1)) for k, s in enumerate(SOURCES)}
    blob = encode_case(pos, fields, scale)
    assert len(blob) == n * 2 * 2 + len(SOURCES) * c * n * 2
    f = np.frombuffer(blob[n * 4:], dtype="<f2").reshape(len(SOURCES), c, n)
    # every value is (k+1) after dividing by its channel scale
    for k in range(len(SOURCES)):
        assert np.allclose(f[k], k + 1)


def _airfoil(n_wall=400, seed=0):
    """Ellipse 'airfoil' on x in [0, 1] with a jittered, geometrically layered mesh around it."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n_wall, endpoint=False)
    wall = np.c_[0.5 + 0.5 * np.cos(t), 0.06 * np.sin(t)]
    normal = np.c_[0.06 * np.cos(t), 0.5 * np.sin(t)]
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    layers = np.geomspace(1e-5, 0.1, 60)
    near = (wall[None] + layers[:, None, None] * normal[None]).reshape(-1, 2)
    near += rng.normal(scale=1e-4, size=near.shape) * (layers.repeat(n_wall)[:, None] / 0.1)
    far = rng.uniform([-0.6, -0.7], [2.2, 0.7], size=(20000, 2))
    pos = np.concatenate([wall, near, far])
    surf = np.zeros(len(pos), bool)
    surf[:n_wall] = True
    from scipy.spatial import cKDTree
    dense = np.c_[0.5 + 0.5 * np.cos(np.linspace(0, 2 * np.pi, 20000)), 0.06 * np.sin(np.linspace(0, 2 * np.pi, 20000))]
    wall_dist, _ = cKDTree(dense).query(pos)
    inside = ((pos[:, 0] - 0.5) / 0.5) ** 2 + (pos[:, 1] / 0.06) ** 2 < 1
    keep = ~inside | surf
    return pos[keep], surf[keep], wall_dist[keep]


def test_wall_layers_fill_one_point_per_station_and_level():
    pos, surf, wall_dist = _airfoil()
    idx = wall_layers(pos, surf, wall_dist, stride=4, levels=12)
    assert not surf[idx].any() and (wall_dist[idx] < 0.04).all()
    # 100 stations x 12 levels, give or take empty bins
    assert 0.9 * 1200 <= len(idx) <= 1200


def test_sample_keeps_wall_layers_and_keeps_random_points_off_the_wall():
    pos, surf, wall_dist = _airfoil()
    idx = sample_points(pos, surf, 3000, np.random.default_rng(0), wall_dist=wall_dist)
    layers = wall_layers(pos, surf, wall_dist)
    n_surf = surf.sum()
    assert len(idx) == 3000 and len(np.unique(idx)) == 3000
    assert np.array_equal(idx[n_surf:n_surf + len(layers)], layers)
    assert (wall_dist[idx[n_surf + len(layers):]] >= 0.04).all()
