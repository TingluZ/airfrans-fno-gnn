# AirfRANS surrogate: FNO vs GNN

Comparing a Fourier Neural Operator against a graph neural network (GraphSAGE)
as surrogate models for steady-state RANS airfoil aerodynamics.

**[Explore the results interactively →](https://tingluz.github.io/tinglu-portfolio/projects/airfrans-fno-gnn/)**
Compare CFD, FNO and GNN predictions on any of the 200 test airfoils, zoom into the
boundary layer, and see per-case errors, on [my portfolio site](https://tingluz.github.io/tinglu-portfolio/).

## Problem statement

The traditional way of quantifiying flow field is computational simulation. During my Master's, I used Ansys Fluent to simulate building environment, airplane cabin, human respiration, etc. Sometimes the model could have millions of meshes, and the whole simulation might take several days. If I wanted to change any parameter or boundary condition, I needed to go through the time-consuming process again. The computational simulation is important as it is based on real Physics, however, when we have enough data from the simulations, we could leverage the strength of data driven models. Once those models are trained, it takes much less time to do a new prediction with new inputs.This project is aiming to predict pressure and velocity fields around an airfoil directly from
geometry + flow conditions, skipping the iterative CFD solve.

## Dataset

This project use AirfRANS (Bonnet et al., 2022) dataset, a dataset for studying two-dimentional incompressible steady-state Reynolds-Averaged Navier-Stokes equations over airfoils at a sub-sonic regime and for different angles of attack. There are 800 samples for the training set and 200 samples for the test set. The features (inputs) are: x and y coordinate of mesh points, x and y component of inlet velocity, distance to airfoil and normal x, y component. The targets (outputs) are: x and component of velocity, pressure and turbulent viscosity.

Source: 

https://github.com/Extrality/AirfRANS

https://arxiv.org/abs/2212.07564


## Method

**FNO** — interpolates node features onto a regular grid, applies
Fourier layers (spectral convolution in truncated Fourier space), then projects
back to output channels.

**GNN** — a 4-layer GraphSAGE stack (PyTorch Geometric `SAGEConv`, 32 hidden channels,
~5k parameters) with one node per mesh point. In the
results below, each node is connected to its 8 nearest neighbours (k-NN). Edges carry
connectivity only, no edge features.

The graph can alternatively be built from the real CFD mesh (`--graph mesh`): AirfRANS
meshes are all-quadrilateral, so the edges are the four sides of every cell
(~725k directed edges per simulation vs. ~1.65M for k-NN, and ~30% cheaper to train).
Its results are in the graph ablation below.

## Results

All numbers below are from the AirfRANS test set.  Validation-set numbers were checked against these and matched within a few percentage points across every channel, confirming the tuning process didn't overfit to val.

**Model summary**

| Model | Parameters | Training time (50 epochs) | Inference time (ms/sim) |
|-------|-----------:|---------------------------:|-------------------------:|
| FNO   | 4.34M      | ~35 min (MPS)              | ~1856 ms                 |
| GNN (SAGE, k-NN) | 5k  | ~8.2 hrs (CPU)             | ~707 ms                  |

FNO trains far faster but infers slower — its inference cost is dominated
by Delaunay triangulation and grid interpolation over ~180k mesh points. 
GNN's training is the expensive side as a k-NN graph is rebuilt every epoch, every simulation,
but inference is cheaper: a smaller model plus a faster k-NN build.

**Point-space relative L2 error, by target channel**

| Target   | FNO (all points) | GNN (all points) | FNO (surface) | GNN (surface) |
|----------|------------------:|------------------:|---------------:|---------------:|
| vx       | 36.8%             | 22.1%             | 115.5%         | 19.0%          |
| vy       | 60.2%             | 37.9%             | 66.7%          | 19.4%          |
| pressure | 85.1%             | 56.7%             | 80.5%          | 53.7%          |
| nu_t     | 61.1%             | 49.5%             | 53.6%          | 19.4%          |

*(GNN = SAGEConv on the k-NN graph.)*

GNN wins every channel, in both splits — especially at the surface
, where FNO's fixed-resolution grid structurally cannot
resolve the boundary layer. Pressure is the hardest channel for both models; its error
distribution is heavy-tailed (a handful of high-angle-of-attack
simulations dominate the mean — see `comparison.ipynb`), not uniformly
poor across the test set.

## Discussion

FNO requires interpolating features from points to a regular grid, then interpolating
the grid output back to points. This point-grid-point transformation introduces an error
floor: even with a perfect model, no amount of training can push the error below what this
round trip alone costs. I measured this directly, before training any model, by round-tripping
the true fields through the grid and back:

| Grid res  | vx (all pts) | pressure (all pts) | vx (surface) | pressure (surface) |
|-----------|-------------:|--------------------:|--------------:|---------------------:|
| 128×64    | 49.3%        | 26.1%                | 117.9%        | 62.3%                 |
| 256×128   | 32.3%        | 13.0%                | 100.7%        | 34.3%                 |
| 512×256   | 22.1%        | 10.2%                |  81.0%        | 30.2%                 |

*(relative MAE / std, averaged over 3 training simulations)*

256×128 was chosen because it's the point of diminishing returns: 128×64→256×128 roughly
halves the error, but 256×128→512×256 costs 4× the disk space for a much smaller gain.
More importantly, the **surface** floor barely moves with resolution at all. `n_modes=(32, 64)` was chosen to match the
grid's 2:1 aspect ratio, so the FNO resolves the same physical wavelength in both directions.

Also, the regular grid makes FNO struggle at the surface. During CFD simulation, the mesh needs to be much denser near the surface than far field, in order to better simulate the complex boundary layer. When interpolating these mesh points into a regular grid, there is more information lost near the surface. While there is a trade-off on the grid resolution, as discussed before, we also need to consider the importance of the grid cells near the surface. Therefore a density-weighted loss is introduced, weighting each grid cell's contribution
to the training loss by how many real mesh points map into it. Measured on the validation
split, before vs. after:

| Target   | all pts (before) | all pts (after) | surface (before) | surface (after) |
|----------|------------------:|------------------:|--------------------:|-------------------:|
| vx       | 43.6%             | 37.7%             | 126.3%              | 117.8%             |
| vy       | 79.3%             | 63.0%             | 32.0%               | 62.7%              |
| pressure | 78.4%             | 85.2%             | 94.0%               | 86.7%              |
| nu_t     | 64.7%             | 58.1%             | 55.1%               | 53.0%              |

Three of four channels improved. Pressure got slightly worse overall despite improving at
the surface, and `vy` at the surface got clearly worse even though its own all-points number
improved. A single shared weight map applied identically across all four output channels
can't selectively help one channel without affecting how the others are learned through the
same underlying grid representation.

**Graph ablation (SAGE on k-NN vs. SAGE on the real mesh).** Same model, only the edges differ. Relative L2 error on the 80 validation simulations
(mean over simulations):

| Target   | all pts, k-NN | all pts, mesh | surface, k-NN | surface, mesh |
|----------|--------------:|--------------:|--------------:|--------------:|
| vx       | 22.4%         | 25.3%         | 18.8%         | 19.9%         |
| vy       | 38.2%         | 43.1%         | 19.7%         | 17.7%         |
| pressure | 58.8%         | 63.2%         | 60.7%         | 64.8%         |
| nu_t     | 47.1%         | 50.1%         | 19.3%         | 13.3%         |

Best validation loss was 0.274 (k-NN) vs. 0.294 (mesh). The real mesh is not better
overall. It is 3 to 5 points worse on every all-points channel and on surface pressure,
and better on surface `vy` and especially surface `nu_t`. Since the mesh graph has
about 2.3x fewer edges and a degree of 4 rather than 8, one plausible reason is its
smaller receptive field with only 4 layers, but I did not test that. These are single runs
with one seed, and the per-simulation spread (e.g. +/-6% on `vx`) is comparable to several
of the gaps, so only the larger ones (surface `nu_t`, all-points `vy`) should be read as signal.

Compared with FNO, GNN's nodes are the mesh points themselves, there is no need for a point-grid-point transformation, so GNN wins mechanistically, and this can clearly be seen from the L2 error comparison. But GNN costs more than 10x on training time. Nonetheless, GNN also struggles on pressure prediction, correlated with angle of attack. The challenge is in physics, not model architecture.

Therefore I also tried adding a physics constraint directly: a no-slip penalty on GNN
training, penalising predicted velocity at surface nodes (which is exactly zero by the
no-slip boundary condition). Measured directly at the metric this penalty targets, after
15 epochs on a 100-simulation subset:

| | mean \|v\| at the wall (m/s) | surface vx, vy MSE |
|---|---:|---:|
| without penalty | 17.4 | 213.6 |
| with penalty (weight 0.1) | 42.4 | 910.6 |

The penalty made the targeted quantity more than twice as bad, and a sweep of
weights (0.1 to 100) showed held-out accuracy degrading as the weight
increased, and the largest weight visibly failing to improve between epochs. The cause could be that all four output channels are produced from the same shared message-passing
layers, so a loss term that only touches 2 channels at ~0.5% of nodes creates gradient
interference in that shared representation rather than a clean, isolated improvement. The
penalty was not used in the final GNN for this reason.

## Limitations

- **Single train/val/test split, no cross-validation.** All reported numbers are point
  estimates with no error bars or confidence intervals.
- **The headline GNN uses a k-NN graph (k=8), not the true mesh connectivity.** The
  mesh-graph variant (`--graph mesh`) was trained and gave similar overall accuracy (see the
  graph ablation), but with a single seed and no depth sweep, so the effect of the graph
  choice is only roughly measured.
- **No edge features.** The GNN uses bare connectivity only, no relative position or
  distance on edges.
- **Limited hyperparameter search.** Grid resolution was chosen from a small, quick sweep
  on 3 simulations; neither model's architecture hyperparameters (hidden channels, depth,
  learning rate) were swept systematically.
- **The no-slip penalty was only tested on the GNN**, at one narrow weight range, on a
  100-simulation subset for 15 epochs. It was not validated at full scale, and not tried on the FNO.
- **No uncertainty quantification.** Both models produce point predictions only.

## Future work

- A geometry-aware FNO variant (GeoFNO, U-FNO) to close the grid-representation gap
  directly, rather than working around it with a density-weighted loss.
- Add edge features (relative position, distance) and edge/node update blocks on the mesh
  graph for a MeshGraphNet-faithful model, keeping the two SAGEConv models as baselines.
  The mesh graph has a smaller receptive field than k-NN, so this likely needs more
  message-passing blocks than the current 4.
- Per-channel loss weighting, rather than one weight map/penalty shared identically across
  all four outputs, to avoid the cross-channel interference seen in both the density-weighted
  FNO (`vy` regression) and the no-slip penalty (negative result overall).
- A properly scoped no-slip (or divergence-free) constraint experiment: full-scale
  training, a wider weight sweep, and a warmup schedule that introduces the penalty only
  after the main objective has stabilised.
- Investigate the angle-of-attack correlation further — e.g. reweighted sampling of
  high-AoA simulations during training, since that's the identified driver of the
  pressure-prediction difficulty shared by both models.
- Uncertainty quantification via ensembling.
- Evaluate on airfoil shapes genuinely outside the training distribution, not just a
  held-out split of the same shape family.


## Quick start

```bash
pip install -e .

# build the FNO's grid cache + normaliser (run once, ~38 min)
python scripts/build_cache.py

# train both baselines
python scripts/train.py          # FNO, ~35 min on MPS
python scripts/train_gnn.py      # GNN (SAGE on k-NN), ~8 hrs on CPU
python scripts/train_gnn.py --graph mesh   # same model on the real mesh graph, ~3-4 hrs on CPU

# evaluate against the held-out val split
python scripts/evaluate.py
python scripts/evaluate_gnn.py             # add --graph mesh for the mesh variant

# head-to-head comparison, training curves, qualitative plots
jupyter notebook comparison.ipynb

# final numbers on the true held-out test set (run once)
jupyter notebook run_testset.ipynb

# export test-set predictions for the web explorer on the portfolio site
python scripts/export_web.py --out ../tinglu-portfolio/public/data/airfrans

# run the test suite
pytest tests/
```

## Repository structure

```
airfrans-fno-gnn/
├── src/afrans/              # the package
│   ├── config.py            # single source of truth: paths, column layout, GridConfig/TrainConfig
│   ├── splits.py            # train/val split + simulation name lookup
│   ├── interpolate.py       # point <-> grid interpolation (FNO)
│   ├── cache.py             # FNO grid cache + density-weight-map construction
│   ├── grid_dataset.py      # FNO's Dataset (from_cache / from_dataset)
│   ├── graph_dataset.py     # GNN graph construction (k-NN or real mesh connectivity)
│   ├── normalise.py         # Normaliser + streaming RunningStats
│   ├── metrics.py           # point-space relative L2
│   ├── engine.py            # train/val loops, checkpointing, both models
│   ├── web_export.py        # cropping, sampling and binary encoding for the web explorer
│   └── models/
│       ├── fno.py
│       └── gnn.py
├── scripts/
│   ├── build_cache.py       # build the FNO's grid cache (once per resolution)
│   ├── train.py             # train the FNO
│   ├── train_gnn.py         # train the GNN
│   ├── evaluate.py          # FNO point-space evaluation
│   ├── evaluate_gnn.py      # GNN point-space evaluation
│   └── export_web.py        # test-set predictions -> static files for the portfolio explorer
├── tests/                   # pytest -- covers several real bugs found during development
├── initial_investigation.ipynb   # first dataset exploration
├── fno_investigation.ipynb       # diagnosing the FNO's training-objective mismatch
├── comparison.ipynb              # FNO vs GNN: training curves, timing, accuracy, qualitative plots
├── run_testset.ipynb             # final numbers on the true held-out test set
├── results.json / results_gnn.json / results_test.json   # saved evaluation outputs
├── checkpoints/              # saved model weights (gitignored)
└── data/                     # AirfRANS dataset + grid/weight caches (gitignored)
```
