import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from afrans.serving import make_fixture  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    root = make_fixture(tmp_path_factory.mktemp("artifacts"))
    monkeypatch_module.setenv("ARTIFACT_DIR", str(root))
    from serve.app import app
    with TestClient(app) as c:  # the with-block runs the startup that loads the model
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["n_cases"] == 2


def test_cases_lists_ids_and_limits(client):
    body = client.get("/cases").json()
    assert [c["id"] for c in body["cases"]] == [0, 1]
    assert body["limits"]["aoa_deg"] == [-5.0, 15.0]


def test_predict_at_own_inflow_returns_truth_and_error(client):
    body = client.post("/predict", json={"case_id": 0, "n_points": 1000}).json()
    assert body["in_dataset"] is True
    assert body["truth"] is not None and body["error"] is not None
    n = len(body["x"])
    assert 0 < n <= 1000
    assert all(len(body["prediction"][c]) == n for c in ("vx", "vy", "pressure", "nu_t"))


def test_predict_what_if_has_no_truth(client):
    body = client.post("/predict", json={"case_id": 0, "aoa_deg": 12.5, "u_inf": 60}).json()
    assert body["in_dataset"] is False
    assert body["truth"] is None and body["error"] is None


@pytest.mark.parametrize("payload", [
    {"case_id": 0, "aoa_deg": 30},       # outside the training range
    {"case_id": 0, "u_inf": 5},
    {"case_id": 0, "n_points": 10},
])
def test_out_of_range_inputs_are_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_unknown_case_is_404(client):
    assert client.post("/predict", json={"case_id": 999}).status_code == 404
