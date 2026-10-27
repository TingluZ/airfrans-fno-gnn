"""
Point-space error metrics -- physical units

Grid space MSE ( what fit() trains on and reports)
is training signal, not a comparable result:
FNO and GNN don't share a grid so only point-space
numbers can go in the same table
"""

import numpy as np

def relative_l2(pred, target, mask=None, ref_norm=None, eps=1e-8):
    """
    Per-channel ||pred - target||_2 / ||target||_2, shape (C, )

    Ref defaults to ||target||_2 over the same points
    
    Standard neural-operator convention (Li et al., FNO): each simulation's
    error is scaled by that simulation's own norm, not a global one, and the
    per-sim values are averaged afterward.
    If mask is given, both the error and the norm are computed over the masked
    subset only.  

    At surface, velocity and nu_t are exactly 0 by the no-slip condition.
    Pass a fixed ref_norm (e.g. the training normaliser's global std) to measure
    error relative to the field's normal scale instead. The evaluation scripts pass
    target_std * sqrt(n_masked), which makes the result RMSE / target_std.
    """

    if mask is not None:
        pred, target = pred[mask], target[mask]
    denom = ref_norm if ref_norm is not None else np.linalg.norm(target, axis=0)
    return np.linalg.norm(pred - target, axis=0) / (denom + eps)

def summarize(per_sim_errors, names):
    """
    List of (C,) arrays, one per simulation -> {name: {mean, std}} across sims
    """

    arr = np.stack(per_sim_errors)
    return {
        name: {"mean": float(arr[:, j].mean()), "std":float(arr[:, j].std())}
        for j, name in enumerate(names)
    }