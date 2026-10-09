"""
Builds a neuralop FNO from TrainConfig
"""

from neuralop.models import FNO
from afrans.config import N_FEATURES, N_TARGETS

def build_fno(train_cfg):
    return FNO(
        n_modes = train_cfg.n_modes,
        in_channels = N_FEATURES,
        out_channels = N_TARGETS,
        hidden_channels = train_cfg.hidden_channels,
        n_layers = train_cfg.n_layers 
    )