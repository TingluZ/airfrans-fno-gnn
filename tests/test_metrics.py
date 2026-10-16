import numpy as np

from afrans.metrics import relative_l2, summarize


def test_relative_l2_perfect_prediction_is_zero():
    target = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert np.allclose(relative_l2(target, target), 0.0)


def test_relative_l2_mask_restricts_to_subset():
    pred = np.array([[0.0], [0.0], [10.0]])
    target = np.array([[0.0], [0.0], [0.0]])
    mask = np.array([True, True, False])  # excludes the one bad point

    # masked out the only error -> perfect score
    assert relative_l2(pred, target, mask=mask, ref_norm=np.array([1.0]))[0] == 0.0


def test_relative_l2_zero_target_blows_up_without_ref_norm():
    """Reproduces the exact no-slip bug: velocity/nu_t are exactly zero at
    the surface, so the default (local target norm) denominator collapses
    and the ratio explodes for any nonzero prediction."""
    pred = np.full((5, 1), 0.01)
    target = np.zeros((5, 1))

    result = relative_l2(pred, target)

    assert result[0] > 1e4


def test_relative_l2_ref_norm_avoids_the_blowup():
    """Passing a fixed, physically meaningful scale (e.g. the normaliser's
    global std) in place of the local target norm keeps the same inputs
    from exploding."""
    pred = np.full((5, 1), 0.01)
    target = np.zeros((5, 1))

    result = relative_l2(pred, target, ref_norm=np.array([1.0]))

    assert result[0] < 1.0


def test_summarize_matches_plain_numpy():
    per_sim = [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.2, 0.6])]
    names = ["a", "b"]

    out = summarize(per_sim, names)
    arr = np.stack(per_sim)

    assert np.isclose(out["a"]["mean"], arr[:, 0].mean())
    assert np.isclose(out["a"]["std"], arr[:, 0].std())
    assert np.isclose(out["b"]["mean"], arr[:, 1].mean())
    assert np.isclose(out["b"]["std"], arr[:, 1].std())
