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


def wall_layers(pos, surf, wall_dist, stride=4, levels=12, d_min=2e-4, band=0.04, box=BOX):
    """
    Indices of points arranged in layers along the wall: for every `stride`-th
    surface point (a station) and each of `levels` wall distances spaced
    geometrically from d_min to band, the mesh point nearest the station's
    normal line within that distance bin

    The boundary layer is far thinner than random samples are spaced, so
    random points near the wall land some inside it and some outside and the
    browser's triangulation draws spikes. Layers that follow the wall keep
    each wall distance together, so the boundary layer renders as smooth bands.
    """
    s_idx = np.flatnonzero(surf & in_box(pos, box))
    cand = np.flatnonzero(~surf & in_box(pos, box) & (wall_dist < band))
    if not len(s_idx) or not len(cand):
        return np.array([], dtype=int)
    # order the surface around the airfoil (chord along x in [0, 1]) and keep every stride-th
    ang = np.arctan2(pos[s_idx, 1], pos[s_idx, 0] - 0.5)
    stations = s_idx[np.argsort(ang)][::stride]

    dist, j = cKDTree(pos[stations]).query(pos[cand])
    d = np.maximum(wall_dist[cand], 1e-12)
    lateral = np.sqrt(np.maximum(dist ** 2 - d ** 2, 0))
    edges = np.geomspace(d_min, band, levels + 1)
    k = np.searchsorted(edges, d) - 1
    ok = (k >= 0) & (k < levels)
    cand, j, k, lateral = cand[ok], j[ok], k[ok], lateral[ok]

    # per (station, level) bin, the candidate closest to the normal line
    order = np.lexsort((lateral, k, j))
    cell = j[order] * levels + k[order]
    first = np.r_[True, cell[1:] != cell[:-1]]
    return np.sort(cand[order][first])


def sample_points(pos, surf, n_points, rng, wall_dist=None, box=BOX, k=6, band=0.04):
    """
    Indices to keep: all surface points in the box first, then (given
    wall_dist) the wall layers from wall_layers(), then the rest of the
    n_points budget sampled away from the wall, half by mesh density and
    half by local cell area (so the far field isn't empty)
    """
    keep_surf = np.flatnonzero(surf & in_box(pos, box))
    away = ~surf & in_box(pos, box)
    layers = np.array([], dtype=int)
    if wall_dist is not None:
        layers = wall_layers(pos, surf, wall_dist, band=band, box=box)
        away &= wall_dist >= band
    cand = np.flatnonzero(away)
    n_rest = max(n_points - len(keep_surf) - len(layers), 0)
    if n_rest >= len(cand):
        return np.concatenate([keep_surf, layers, cand])

    # distance to the k-th neighbour ~ local cell size; squared ~ cell area
    d, _ = cKDTree(pos[cand]).query(pos[cand], k=k + 1)
    area = d[:, -1] ** 2
    p = 1 / len(cand) + area / area.sum()
    picked = rng.choice(cand, size=n_rest, replace=False, p=p / p.sum())
    return np.concatenate([keep_surf, layers, np.sort(picked)])


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
