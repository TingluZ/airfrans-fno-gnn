"""
Build the grid caches and the normaliser
Run once per GridConfig

python scripts/build_cache.py --limit 5 # smoke test
python scripts/build_cache.py
"""

import argparse
import time

import airfrans as af

from afrans.cache import build_grid_cache
from afrans.config import CACHE_DIR, DATA_ROOT, FEATURES, TARGETS, GridConfig
from afrans.normalise import Normaliser, RunningStats
from afrans.splits import get_split

def fit_normaliser(dataset, indices, verbose=True):
    """
    Streaming fit over the TRAIN simulations only, in point space
    """
    f_stats, t_stats = RunningStats(), RunningStats()
    for k, i in enumerate(indices):
        sim = dataset[i]
        f_stats.update(sim[:, FEATURES])
        t_stats.update(sim[:, TARGETS])
        if verbose and (k + 1) % 100 == 0:
            print(f" normaliser {k + 1}/{len(indices)}")

    norm = Normaliser()
    norm.fit_from_stats(*f_stats.result(), *t_stats.result())
    return norm

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="full")
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N train sims; writes to *_smoke paths")
    p.add_argument("--grid_res", type=int, nargs=2, metavar=("NX", "NY"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = (GridConfig() if args.grid_res is None
           else GridConfig(grid_res_x=args.grid_res[0], grid_res_y=args.grid_res[1]))

    # a smoke run must never be able to clobber a real cache
    suffix = "_smoke" if args.limit else ""

    print(f"grid {cfg.tag}, task {args.task}{suffix}")
    print("loading dataset (~2.5min, 6.5 GB) ...")
    t0 = time.perf_counter()
    dataset, names = af.dataset.load(root=str(DATA_ROOT), task=args.task, train=True)
    print(f" loaded {len(dataset)} sims in {time.perf_counter() - t0:.0f} s")

    split = get_split(names=names, task=args.task)
    train_idx, val_idx = split["train"], split["val"]
    if args.limit:
        train_idx = train_idx[:args.limit]
        val_idx = val_idx[:max(1, args.limit //9)]
    print(f"train {len(train_idx)} val {len(val_idx)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    norm = fit_normaliser(dataset, train_idx)
    norm.save(CACHE_DIR / f"normaliser{suffix}.npz")
    print(f"normaliser fitted in {time.perf_counter() - t0:.0f} s")

    for name, idx in (("train", train_idx), ("val", val_idx)):
        t0 = time.perf_counter()
        build_grid_cache(dataset, idx, cfg.cache_path(name + suffix),
                         cfg, overwrite=args.overwrite)
        dt = time.perf_counter() - t0
        print(f"{name}: {dt:.0f} s for {len(idx)} sims "
              f"({dt / len(idx):.2f} s/sim -> {dt / len(idx) * 800 / 60: .0f} min for 800)")

if __name__ == "__main__":
    main()