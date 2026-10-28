import numpy as np
import pytest

from afrans.config import TrainConfig
from afrans.graph_dataset import node_features
from afrans.inference import load_gnn, predict_case
from afrans.serving import load_case, load_manifest, make_fixture


def test_node_features_match_airfrans_column_order():
    """x, y, u_xin, u_yin, distance, nx, ny -- the order build_graph and the
    normaliser were built on. A swapped column would silently give garbage."""
    pos = np.array([[0.1, 0.2], [0.3, 0.4]])
    f = node_features(pos, np.array([[0.5], [0.6]]), np.array([[1.0, 0.0], [0.0, 1.0]]),
                      aoa=np.radians(90), u_inf=10.0)
    assert f.shape == (2, 7)
    np.testing.assert_allclose(f[:, 0:2], pos)
    np.testing.assert_allclose(f[:, 2:4], [[0, 10], [0, 10]], atol=1e-12)
    np.testing.assert_allclose(f[:, 4], [0.5, 0.6])
    np.testing.assert_allclose(f[:, 5:7], [[1, 0], [0, 1]])


def test_sdf_shape_does_not_matter():
    """AirfRANS gives sdf as (N, 1); the serving artifacts store it as (N,)."""
    pos, normals = np.zeros((3, 2)), np.zeros((3, 2))
    a = node_features(pos, np.ones((3, 1)), normals, 0.1, 40.0)
    b = node_features(pos, np.ones(3), normals, 0.1, 40.0)
    np.testing.assert_array_equal(a, b)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    return make_fixture(tmp_path_factory.mktemp("artifacts"))


def test_predict_case_shape_and_inflow_dependence(artifacts):
    model, cfg, norm = load_gnn(artifacts / "model.pt", artifacts / "normaliser.npz")
    case = load_case(artifacts / load_manifest(artifacts)[0]["file"])
    args = (model, cfg, norm, case["pos"], case["sdf"], case["normals"])
    a = predict_case(*args, aoa=np.radians(2), u_inf=50.0)
    b = predict_case(*args, aoa=np.radians(10), u_inf=50.0)
    assert a.shape == (len(case["pos"]), 4)
    assert np.isfinite(a).all()
    assert not np.allclose(a, b), "changing the angle of attack must change the prediction"


def test_mesh_graph_model_is_refused_without_edges(artifacts):
    model, _, norm = load_gnn(artifacts / "model.pt", artifacts / "normaliser.npz")
    case = load_case(artifacts / load_manifest(artifacts)[0]["file"])
    with pytest.raises(ValueError, match="k-NN"):
        predict_case(model, TrainConfig(model="gnn", graph="mesh"), norm,
                     case["pos"], case["sdf"], case["normals"], 0.0, 50.0)
