"""Single source of truth for paths, dataset layout, and run configuration."""

from dataclasses import dataclass, fields, asdict
from pathlib import Path

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "Dataset"
CACHE_DIR = REPO_ROOT / "data" / "cache"
CKPT_DIR = REPO_ROOT / "checkpoints"
SPLIT_FILE = REPO_ROOT / "data" / "split.json"

# ── AirfRANS column layout (fixed by the dataset format, not a choice) ────────
COLS = [
    "x", "y",                # 0:2   position
    "u_xin", "u_yin",        # 2:4   inlet velocity
    "distance",              # 4     signed distance to airfoil
    "nx", "ny",              # 5:7   surface normals
    "vx", "vy",              # 7:9   velocity          (target)
    "pressure",              # 9     kinematic pressure (target)
    "nu_t",                  # 10    turbulent viscosity (target)
    "surface",               # 11    surface indicator mask
]

POS = slice(0, 2)
FEATURES = slice(0, 7)
TARGETS = slice(7, 11)
SURFACE = 11

N_FEATURES = FEATURES.stop - FEATURES.start   # 7
N_TARGETS = TARGETS.stop - TARGETS.start      # 4
TARGET_NAMES = COLS[TARGETS]


@dataclass(frozen=True)
class GridConfig:
    """Everything that determines the contents of a grid cache file.

    Two caches are interchangeable if and only if their GridConfigs are equal.
    Nothing that leaves the cached array unchanged belongs in here.
    """

    grid_res_x: int = 256
    grid_res_y: int = 128
    x_min: float = -2.0
    x_max: float = 4.0
    y_min: float = -1.5
    y_max: float = 1.5

    @property
    def shape(self):
        """(H, W) — row-major, y first. Write this ordering in exactly one place."""
        return (self.grid_res_y, self.grid_res_x)

    @property
    def extent(self):
        """[left, right, bottom, top] — the order plt.imshow expects."""
        return [self.x_min, self.x_max, self.y_min, self.y_max]

    @property
    def tag(self):
        return f"{self.grid_res_y}x{self.grid_res_x}"

    def mesh(self):
        """Query points for point->grid interpolation, both (H, W)."""
        return np.meshgrid(
            np.linspace(self.x_min, self.x_max, self.grid_res_x),
            np.linspace(self.y_min, self.y_max, self.grid_res_y),
        )

    def axes(self):
        """1-D (y_coords, x_coords) — the order RegularGridInterpolator expects."""
        return (
            np.linspace(self.y_min, self.y_max, self.grid_res_y),
            np.linspace(self.x_min, self.x_max, self.grid_res_x),
        )

    def cache_path(self, split):
        return CACHE_DIR / f"grids_{split}_{self.tag}.npz"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_mapping(cls, m):
        """Rebuild from an np.load()'d npz, where scalars come back as 0-d arrays."""
        kwargs = {}
        for f in fields(cls):
            v = m[f.name]
            v = v.item() if hasattr(v, "item") else v
            kwargs[f.name] = int(v) if f.type is int else float(v)
        return cls(**kwargs)


@dataclass(frozen=True)
class TrainConfig:
    """Model and optimisation choices. Changing any of these does NOT
    invalidate a grid cache."""

    model: str = "fno"                # fno | geofno | gnn
    n_modes: tuple = (32, 64)
    hidden_channels: int = 32
    n_layers: int = 4
    k_neighbors: int = 8

    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 4
    epochs: int = 50
    grad_clip: float = 0.0            # 0 disables

    seed: int = 42
    normalise_on: str = "points"        # grid | points — record which regime a run used
    loss_weighting: str = "none"        # none | density - record which training objective was used
    device: str = "auto"              # resolved to cuda/mps/cpu in engine.py
    num_workers: int = 0
    no_slip_weight: float = 0.0
    tag: str = ""                      # e.g. "smoke" -- throwaway/debug runs, folded into run_name

    def run_name(self, grid_cfg=None):
        extra = f"_{self.tag}" if self.tag else ""
        if self.model =='fno':
            modes = "x".join(str(m) for m in self.n_modes)
            tag = "" if self.loss_weighting == "none" else f"_{self.loss_weighting}"
            return (f"{self.model}_h{self.hidden_channels}_m{modes}"
                    f"_lr{self.lr:g}_s{self.seed}{tag}_{grid_cfg.tag}{extra}")
        ns_tag = "" if self.no_slip_weight == 0 else f"_ns{self.no_slip_weight:g}"
        return f"{self.model}_h{self.hidden_channels}_l{self.n_layers}_k{self.k_neighbors}_lr{self.lr:g}_s{self.seed}{ns_tag}{extra}"

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
            object.__setattr__(self, "n_modes", tuple(self.n_modes))
