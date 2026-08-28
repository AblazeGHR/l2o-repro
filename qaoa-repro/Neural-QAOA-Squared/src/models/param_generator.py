import sys
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import math
import time
from .gat_encoder import GATEncoder
from .gcn_encoder import GCNEncoder
from torch_geometric.nn import global_mean_pool


class ParamGenerator(nn.Module):
    def __init__(self):
        super(ParamGenerator, self).__init__()

        self.partition_encoder = GCNEncoder(
            input_dim=config.NODE_FEATURE_DIM,
            hidden_dim=config.GENERATOR_GNN_HIDDEN_DIM,
            num_layers=config.GENERATOR_GNN_NUM_LAYERS,
        )

        self.aggregator = global_mean_pool

        output_dim = config.QAOA_DEPTH * 2 * 2
        self.mlp = nn.Sequential(
            nn.Linear(config.GENERATOR_GNN_HIDDEN_DIM, config.GENERATOR_MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.GENERATOR_MLP_HIDDEN_DIM, output_dim),
        )

    def forward(self, data):
        x, edge_index_c, edge_weight_c = data.x, data.edge_index_c, data.edge_weight_c
        batch_index = data.subgraph_batch_index

        h_nodes = self.partition_encoder(x, edge_index_c, edge_weight_c)

        h_sub = self.aggregator(h_nodes, batch_index)

        raw_out = self.mlp(h_sub)

        reshaped_out = raw_out.view(-1, config.QAOA_DEPTH * 2, 2)

        normalized_out = F.normalize(reshaped_out, p=2, dim=-1)

        x_comp = normalized_out[:, :, 0]
        y_comp = normalized_out[:, :, 1]

        angles = torch.atan2(y_comp, x_comp)

        P = (angles + 2 * torch.pi) % (2 * torch.pi)

        return P