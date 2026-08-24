"""
Determine train/val split over simulation indices

Load no simulation data but a function of names and a seed
"""

import json
from pathlib import Path

import numpy as np

from afrans.config import DATA_ROOT, SPLIT_FILE

MANIFEST = DATA_ROOT / "manifest.json"

def simulation_names(task="full", train=True, manifest=MANIFEST):
    """Names in the same order af.dataset.load returns simulations."""
    key = f"{task}_{'train' if train else 'test'}"
    with open(manifest) as f:
        m = json.load(f)
    if key not in m:
        raise KeyError(f"{key!r} not in manifest; available: {sorted(m)}")

    return m[key]

def make_split(names, val_frac=0.1, seed=42):
    n = len(names)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n) # a random ordering of positions 0 ... n-1
    n_val = int(round(val_frac * n)) 

    # sorted() so that the JSON is reviewable in a diff
    # and the cache build reads simulation in directory order
    #.tolist because json cannot encode int64
    val_idx = sorted(perm[:n_val].tolist())
    train_idx = sorted(perm[n_val:].tolist())

    return {
        "seed": seed,
        "val_frac": val_frac,
        "n_sims": n,
        "train": train_idx,
        "val": val_idx,
        "train_names": [names[i] for i in train_idx],
        "val_names":[names[i] for i in val_idx],
    }

def save_split(split, path=SPLIT_FILE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(split, indent=2))
    tmp.replace(path)

    return path

def load_split(path=SPLIT_FILE, names=None):
    """
    Load a split. 
    If names is given, indices are re-derived from the 
    stored names so the split survives any change in
    dataset ordering.
    """
    split = json.loads(Path(path).read_text())

    if names is not None:
        position = {name: i for i, name in enumerate(names)}
        for key in ("train", "val"):
            try:
                split[key]=[position[n] for n in split[f"{key}_names"]]
            except KeyError as e:
                raise ValueError(
                    f"{e.args[0]} is in the saved split but not in the current "
                    f"dataset -- wrong task or the dataset changed"
                ) from None

    _validate(split)
    return split

def _validate(split):
    train, val = split["train"], split["val"]
    if set(train) & set(val):
        raise ValueError("train and val overlap")
    if len(train) + len(val) != split["n_sims"]:
        raise ValueError(
            f"split covers {len(train) + len(val)} of {split['n_sims']} simulations"
        )
    

def get_split(names=None, 
              task="full", 
              val_frac=0.1, 
              seed=42, 
              path=SPLIT_FILE):
    """
    Load the split from disk, 
    creating and saving it on first call
    """
    if names is None:
        names = simulation_names(task)
    path = Path(path)
    if path.exists():
        return load_split(path, names)
    split = make_split(names, val_frac, seed)
    save_split(split, path)

    return split