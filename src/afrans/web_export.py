"""
Helpers for exporting predictions to the portfolio web explorer

Kept free of torch so they can be tested without models or data;
scripts/export_web.py does the model runs.
"""

import numpy as np
from scipy.spatial import cKDTree

# Crop window in chord lengths: the airfoil spans x in [0, 1]
BOX = (-0.6, 2.2, -0.7, 0.7)  # x_min, x_max, y_min, y_max
SOURCES = ["truth", "fno", "gnn"]

ATTRIBUTION = {
    "dataset": "AirfRANS",
    "authors": "Florent Bonnet, Jocelyn Ahmed Mazari, Paola Cinnella, Patrick Gallinari",
    "paper": "AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for "
             "Approximating Reynolds-Averaged Navier-Stokes Solutions (NeurIPS 2022 "
             "Datasets and Benchmarks)",
    "paper_url": "https://arxiv.org/abs/2212.07564",
    "data_url": "https://github.com/Extrality/AirfRANS",
    "license": "ODbL-1.0",
    "note": "Derived from AirfRANS: fields are cropped, downsampled and stored at "
            "reduced precision, alongside model predictions.",
}


def in_box(pos, box=BOX):
    x0, x1, y0, y1 = box
    return (pos[:, 0] >= x0) & (pos[:, 0] <= x1) & (pos[:, 1] >= y0) & (pos[:, 1] <= y1)


def sample_points(pos, surf, n_points, rng, box=BOX, k=6, band=0.05):
    """
    Indices to keep: all surface points in the box, plus n_points total
    sampled from the rest, a third each by mesh density, by local cell area,
    and uniformly within `band` chords of the wall

    Mesh density alone puts most samples in the first few cell layers, leaving
    the outer boundary layer sparse; the band stratum fills it so velocity
    gradients near the wall aren't drawn as long slivers.
    """
    keep_surf = np.flatnonzero(surf & in_box(pos, box))
    cand = np.flatnonzero(~surf & in_box(pos, box))
    n_rest = max(n_points - len(keep_surf), 0)
    if n_rest >= len(cand):
        return np.concatenate([keep_surf, cand])

    # distance to the k-th neighbour ~ local cell size; squared ~ cell area
    d, _ = cKDTree(pos[cand]).query(pos[cand], k=k + 1)
    area = d[:, -1] ** 2
    near = np.zeros(len(cand))
    if len(keep_surf):
        wall, _ = cKDTree(pos[keep_surf]).query(pos[cand])
        near = (wall < band).astype(float)
    p = 1 / len(cand) + area / area.sum() + (near / near.sum() if near.any() else 1 / len(cand))
    picked = rng.choice(cand, size=n_rest, replace=False, p=p / p.sum())
    return np.concatenate([keep_surf, np.sort(picked)])


def quantise_pos(pos, box=BOX):
    """Positions -> uint16 over the crop box (~4e-5 chord resolution)."""
    x0, x1, y0, y1 = box
    lo, span = np.array([x0, y0]), np.array([x1 - x0, y1 - y0])
    q = np.round((pos - lo) / span * 65535)
    return np.clip(q, 0, 65535).astype("<u2")


def encode_case(pos, fields, scale, box=BOX):
    """
    One case -> bytes: uint16 positions (n, 2), then float16 fields laid out
    [source][channel][point], each channel divided by its scale so small
    quantities like nu_t keep their precision in float16
    """
    q = quantise_pos(pos, box)
    f = np.stack([fields[s] / scale for s in SOURCES])   # (S, n, C)
    f = np.transpose(f, (0, 2, 1)).astype("<f2")          # (S, C, n)
    return q.tobytes() + f.tobytes()
