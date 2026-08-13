# AirfRANS surrogate: FNO vs GNN

Comparing a Fourier Neural Operator against a mesh-based graph neural network
as surrogate models for steady-state RANS airfoil aerodynamics.

## Problem statement

[Day 6 — fill in: what physical quantity are we predicting and why does it matter?
e.g. predicting pressure and velocity fields around an airfoil directly from
geometry + flow conditions, skipping the iterative CFD solve.]

## Dataset

[Day 6 — fill in: AirfRANS (Bonnet et al., 2022) — cite properly. Note dataset
size, what's in the train/val/test split, what features and targets are used.]

Source: https://github.com/Extrality/AirfRANS

## Method

**FNO** — lifts node features onto a regular grid via interpolation, applies
Fourier layers (spectral convolution in truncated Fourier space), projects
back to output channels.

**GNN** — encode-process-decode architecture inspired by MeshGraphNet
(Pfaff et al., ICLR 2021). Operates directly on the point cloud via a
k-nearest-neighbour graph — no grid interpolation required.

## Results

[Day 6 — paste the table from results.json]

| Model | L2 error (pressure) | L2 error (velocity) | Inference time (ms) |
|-------|---------------------|----------------------|----------------------|
| FNO   |                     |                       |                       |
| GNN   |                     |                       |                       |

## Discussion

[Day 6 — fill in from evaluate.py's worst-case plots: where does each model
fail, and why? Likely candidates: FNO struggles with sharp gradients near
the leading edge due to grid interpolation smoothing; GNN may struggle with
longer-range pressure recovery effects due to limited message-passing radius.]

## Limitations

- FNO requires grid interpolation, which is a real accuracy cost on this
  point-cloud dataset — a geometry-aware variant (GNO, U-FNO) would close
  some of this gap.
- The GNN's k-nearest-neighbour graph is a proxy for true mesh connectivity;
  using the actual triangulation if available would likely help to some degree.
- Single train/val/test split, no cross-validation — error bars not reported.

## Quick start

```bash
pip install -r requirements.txt

# build dataset + split + normalisation (run once)
python dataset.py

# train both baselines
python train_fno.py
python train_gnn.py

# evaluate + compare
python evaluate.py
```

## Repository structure

```
airfrans-fno-gnn/
├── dataset.py       # shared data loading, normalisation, split
├── train_fno.py     # FNO baseline training
├── train_gnn.py     # GNN baseline training
├── evaluate.py       # head-to-head comparison + plots
├── requirements.txt
├── figures/          # saved plots
└── checkpoints/       # saved model weights
```

## Future work

[Day 6 — optional: hyperparameter sweep, uncertainty quantification via
ensembling, geometry-aware FNO variant, extending to held-out airfoil shapes
not seen during training.]
