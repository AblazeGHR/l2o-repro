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


class QueryNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QueryNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class LearnableClusterHead(nn.Module):
    def __init__(self):
        super(LearnableClusterHead, self).__init__()
        self.embed_dim = config.EMBED_DIM
        self.temperature = config.CLUSTER_HEAD_TEMP
        self.learnable_tokens = nn.Parameter(torch.randn(self.embed_dim, self.embed_dim))

        query_input_dim = self.embed_dim

        self.query_net = QueryNet(
            input_dim=query_input_dim,
            hidden_dim=self.embed_dim,
            output_dim=self.embed_dim
        )

    def forward(self, H, k):
        H_norm = F.normalize(H, p=2, dim=1)

        current_tokens = self.learnable_tokens[:k, :]
        query_net_input = current_tokens
        queries = self.query_net(query_net_input)

        queries_norm = F.normalize(queries, p=2, dim=1)

        similarity = torch.matmul(H_norm, queries_norm.T)

        S = F.softmax(similarity / self.temperature, dim=1)

        return S, queries_norm


class OrthogonalComplementHead(nn.Module):
    def __init__(self):
        super(OrthogonalComplementHead, self).__init__()
        self.embed_dim = config.EMBED_DIM
        self.temperature = config.CLUSTER_HEAD_TEMP
        self.register_buffer('global_aux_pool', torch.randn(self.embed_dim, self.embed_dim))

    def forward(self, H, k):
        if k >= self.embed_dim:
            raise ValueError(f"k={k} is too large for embed_dim={self.embed_dim}. Max allowed k is {self.embed_dim - 1}.")

        H_norm = F.normalize(H, p=2, dim=1)
        g = H_norm.mean(dim=0, keepdim=True)
        g = F.normalize(g, p=2, dim=1)

        current_aux = self.global_aux_pool[:k, :]

        matrix_raw = torch.cat([g, current_aux], dim=0)

        Q, _ = torch.linalg.qr(matrix_raw.T)
        Q = Q.T
        queries = Q[1:1 + k]

        similarity = torch.matmul(H_norm, queries.T)
        S = F.softmax(similarity / self.temperature, dim=1)

        return S, queries

class PartitionGenerator(nn.Module):
    def __init__(self):
        super(PartitionGenerator, self).__init__()
        self.topology_encoder = GATEncoder(
            input_dim=config.NODE_FEATURE_DIM,
            hidden_dim=config.GENERATOR_GNN_HIDDEN_DIM,
            num_layers=config.GENERATOR_GNN_NUM_LAYERS,
            edge_dim=config.EDGE_FEATURE_DIM,
        )
        self.cluster_head = OrthogonalComplementHead()

    def forward(self, data, k):
        h_gnn = self.topology_encoder(data.x, data.edge_index, data.edge_attr)
        H = h_gnn
        S, queries = self.cluster_head(H, k)
        return S, H, queries