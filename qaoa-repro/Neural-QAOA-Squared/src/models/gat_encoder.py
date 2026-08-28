import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm


class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, edge_dim):
        super(GATEncoder, self).__init__()
        
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU() 
        )

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim))
            self.norms.append(LayerNorm(hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        x = self.feature_embedding(x)
        
        for i in range(len(self.layers) - 1):
            x_in = x
            
            x = self.layers[i](x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = F.relu(x)
            
            x = x + x_in

        x_in = x
        
        x = self.layers[-1](x, edge_index, edge_attr)
        x = self.norms[-1](x)
        
        x = x + x_in
            
        return x


