import numpy as np

from afrans.normalise import Normaliser, RunningStats


def test_running_stats_matches_numpy_single_batch():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=50.0, scale=0.01, size=(1000, 3))  # std << |mean|, the case this exists for

    stats = RunningStats()
    stats.update(data)
    mean, std = stats.result()

    assert np.allclose(mean, data.mean(axis=0), rtol=1e-6)
    assert np.allclose(std, data.std(axis=0), rtol=1e-4)


def test_running_stats_matches_numpy_across_multiple_batches():
    """The whole point of RunningStats is correctness when the full array
    never exists at once -- so verify the streamed result, built from
    several small .update() calls, matches a one-shot numpy computation
    on the concatenation of all of them."""
    rng = np.random.default_rng(1)
    chunks = [rng.normal(loc=50.0, scale=0.01, size=(n, 2)) for n in (300, 150, 550)]

    stats = RunningStats()
    for chunk in chunks:
        stats.update(chunk)
    mean, std = stats.result()

    full = np.concatenate(chunks, axis=0)
    assert np.allclose(mean, full.mean(axis=0), rtol=1e-6)
    assert np.allclose(std, full.std(axis=0), rtol=1e-4)


def test_normaliser_transform_inverse_roundtrip():
    norm = Normaliser()
    norm.feature_mean = np.array([1.0, -2.0])
    norm.feature_std = np.array([0.5, 2.0])
    norm.target_mean = np.array([10.0])
    norm.target_std = np.array([5.0])

    target = np.array([[12.0], [8.0], [10.0]])
    recovered = norm.inverse_target(norm.transform_target(target))

    assert np.allclose(recovered, target)
