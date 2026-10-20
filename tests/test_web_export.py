import numpy as np

from afrans.web_export import BOX, SOURCES, encode_case, quantise_pos, sample_points


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
