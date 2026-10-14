"""
Bare-connecttivity GNN baseline: 
stacked PyG SAGECinv layers
"""

import torch
from torch_geometric.nn import SAGEConv

from afrans.config import N_FEATURES, N_TARGETS

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, n_layers):
        super().__init__()
        dims = [N_FEATURES] + [hidden_channels] * (n_layers - 1) + [N_TARGETS]
        self.convs = torch.nn.ModuleList(
            SAGEConv(dims[i], dims[i+1]) for i in range(n_layers)
        )
        

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
        return x
        


def build_gnn(train_cfg):
    return GNN(train_cfg.hidden_channels, train_cfg.n_layers)