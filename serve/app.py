"""
"What-if" API for the AirfRANS GraphSAGE surrogate

Pick one of the exported test airfoils, optionally change the angle of attack
and inflow speed, and get the predicted fields on a subsampled set of points.

    uvicorn serve.app:app --port 7860          # from the repo root
    open http://localhost:7860/docs            # interactive docs

Settings (environment variables):
    ARTIFACT_DIR     folder laid out as in afrans.serving (default: ./serving_artifacts)
    HF_MODEL_REPO    if ARTIFACT_DIR has no model.pt, download this Hugging Face repo into it
    ALLOWED_ORIGINS  comma-separated CORS origins (default: the portfolio site)
    TORCH_THREADS    CPU threads for inference (default: 2, what a free Space has)
"""

import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from afrans.config import TARGET_NAMES
from afrans.graph_dataset import knn_edges
from afrans.inference import load_gnn, predict_case
from afrans.metrics import relative_l2
from afrans.serving import AOA_MAX_DEG, AOA_MIN_DEG, U_MAX, U_MIN, load_case, load_manifest
from afrans.web_export import BOX, sample_points

UNITS = {"vx": "m/s", "vy": "m/s", "pressure": "m^2/s^2 (kinematic)", "nu_t": "m^2/s"}
state = {}


def artifact_dir():
    root = Path(os.environ.get("ARTIFACT_DIR", "serving_artifacts"))
    repo = os.environ.get("HF_MODEL_REPO")
    if not (root / "model.pt").exists() and repo:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo, local_dir=root)
    return root


@asynccontextmanager
async def lifespan(app):
    # load once per process, not per request: the model is tiny, but torch
    # import + checkpoint load would otherwise sit on every request's latency
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
    root = artifact_dir()
    model, cfg, norm = load_gnn(root / "model.pt", root / "normaliser.npz")
    cases = {c["id"]: c for c in load_manifest(root)}
    state.update(root=root, model=model, cfg=cfg, norm=norm, cases=cases)
    yield
    state.clear()


app = FastAPI(title="AirfRANS GNN surrogate", version="1.0", lifespan=lifespan,
              description="Predict velocity, pressure and turbulent viscosity around a 2D airfoil "
                          "with a GraphSAGE surrogate trained on AirfRANS. "
                          "Code: https://github.com/TingluZ/airfrans-fno-gnn")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "https://tingluz.github.io").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=8)
def case_with_edges(case_id):
    """The k-NN graph depends only on the mesh, so it's built once per airfoil
    (~0.4 s) and reused for every inflow asked about it."""
    case = load_case(state["root"] / state["cases"][case_id]["file"])
    case["edge_index"] = knn_edges(case["pos"], state["cfg"].k_neighbors)
    return case


class PredictRequest(BaseModel):
    case_id: int = Field(description="an id from GET /cases")
    aoa_deg: Optional[float] = Field(None, ge=AOA_MIN_DEG, le=AOA_MAX_DEG,
                                     description="angle of attack in degrees; default: the case's own")
    u_inf: Optional[float] = Field(None, ge=round(U_MIN, 1), le=round(U_MAX, 1),
                                   description="inflow speed in m/s; default: the case's own")
    n_points: int = Field(7000, ge=500, le=20000, description="points returned, surface first")


@app.get("/")
def root():
    return {"docs": "/docs", "cases": "/cases", "predict": "POST /predict"}


@app.get("/health")
def health():
    return {"status": "ok", "model": state["cfg"].run_name(), "n_cases": len(state["cases"])}


@app.get("/cases")
def cases():
    return {
        "cases": list(state["cases"].values()),
        "limits": {"aoa_deg": [AOA_MIN_DEG, AOA_MAX_DEG], "u_inf": [round(U_MIN, 1), round(U_MAX, 1)]},
    }


@app.post("/predict")
def predict(req: PredictRequest):
    meta = state["cases"].get(req.case_id)
    if meta is None:
        raise HTTPException(404, f"unknown case_id {req.case_id}; see GET /cases")
    aoa_deg = meta["aoa_deg"] if req.aoa_deg is None else req.aoa_deg
    u_inf = meta["u_inf"] if req.u_inf is None else req.u_inf

    t0 = time.perf_counter()
    case = case_with_edges(req.case_id)
    pred = predict_case(state["model"], state["cfg"], state["norm"], case["pos"], case["sdf"],
                        case["normals"], np.radians(aoa_deg), u_inf, edge_index=case["edge_index"])
    ms = (time.perf_counter() - t0) * 1000

    # the CFD answer only exists at the case's own inflow
    at_truth = np.isclose(aoa_deg, meta["aoa_deg"], atol=1e-3) and np.isclose(u_inf, meta["u_inf"], atol=1e-3)
    error = None
    if at_truth:
        surf = case["surface"]
        error = {
            "relative_l2_all": dict(zip(TARGET_NAMES, relative_l2(pred, case["truth"]).round(4).tolist())),
            "surface_rmse_over_std": dict(zip(TARGET_NAMES, relative_l2(
                pred, case["truth"], mask=surf,
                ref_norm=state["norm"].target_std * np.sqrt(surf.sum())).round(4).tolist())),
        }

    idx = sample_points(case["pos"], case["surface"], req.n_points,
                        np.random.default_rng(req.case_id), wall_dist=np.abs(case["sdf"]))
    sig = lambda a: [float(f"{v:.5g}") for v in a]  # 5 significant figures keeps the JSON small
    return {
        "case_id": req.case_id,
        "name": meta["name"],
        "aoa_deg": aoa_deg,
        "u_inf": u_inf,
        "in_dataset": bool(at_truth),
        "inference_ms": round(ms, 1),
        "box": list(BOX),
        "units": UNITS,
        "n_surface": int(case["surface"][idx].sum()),
        "x": sig(case["pos"][idx, 0]),
        "y": sig(case["pos"][idx, 1]),
        "prediction": {name: sig(pred[idx, j]) for j, name in enumerate(TARGET_NAMES)},
        "truth": ({name: sig(case["truth"][idx, j]) for j, name in enumerate(TARGET_NAMES)}
                  if at_truth else None),
        "error": error,
    }
