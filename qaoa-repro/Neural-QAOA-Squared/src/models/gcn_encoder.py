import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import scatter as scatter_add


class AbsNormGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(AbsNormGCNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)

        loop_edge_index, loop_edge_weight = add_self_loops(
            edge_index,
            edge_weight,
            fill_value=1.0,
            num_nodes=x.size(0),
        )
        row, col = loop_edge_index
        deg = scatter_add(loop_edge_weight.abs(), row, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x = self.lin(x)
        return self.propagate(loop_edge_index, x=x, edge_weight=loop_edge_weight, norm=norm)

    def message(self, x_j, edge_weight, norm):
        return (norm.view(-1, 1) * edge_weight.view(-1, 1)) * x_j
    
    def update(self, aggr_out):
        return aggr_out

class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GCNEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, normalize=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, normalize=False))

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)

        x = self.layers[-1](x, edge_index, edge_weight=edge_weight)

        return x




