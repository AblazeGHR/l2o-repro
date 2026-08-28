import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from .gat_encoder import GATEncoder
from .gcn_encoder import GCNEncoder


class Critic_R(nn.Module):
    def __init__(self):
        super(Critic_R, self).__init__()

        self.topology_encoder = GATEncoder(
            input_dim=config.NODE_FEATURE_DIM,
            hidden_dim=config.GNN_HIDDEN_DIM,
            num_layers=config.GNN_NUM_LAYERS,
            edge_dim=config.EDGE_FEATURE_DIM,
        )
        self.partition_encoder = GCNEncoder(
            input_dim=config.NODE_FEATURE_DIM,
            hidden_dim=config.GNN_HIDDEN_DIM,
            num_layers=config.GNN_NUM_LAYERS,
        )
        self.quantum_params_encoder = GATEncoder(
            input_dim=config.QAOA_DEPTH * 2 * 2,
            hidden_dim=config.GNN_HIDDEN_DIM,
            num_layers=config.GNN_NUM_LAYERS,
            edge_dim=config.EDGE_FEATURE_DIM,
        )
        self.aggregator = global_mean_pool

        mlp_input_dim = config.GNN_HIDDEN_DIM * 3

        self.prediction_head = nn.Sequential(
            nn.Linear(mlp_input_dim, config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(config.MLP_HIDDEN_DIM, 1),
        )

    def forward(self, data):
        x, batch = data.x, data.batch
        if hasattr(data, 'h_graph_nodes_cached') and data.h_graph_nodes_cached is not None:
            h_graph_nodes = data.h_graph_nodes_cached
        else:
            edge_index, edge_attr = data.edge_index, data.edge_attr
            if edge_index is None or edge_attr is None:
                 raise ValueError(
                     "Critic_R: Missing edge_index or edge_attr, and h_graph_nodes_cached was not provided."
                 )
            h_graph_nodes = self.topology_encoder(x, edge_index, edge_attr)

        h_graph_agg = self.aggregator(h_graph_nodes, batch)

        edge_index_c, edge_weight_c = data.edge_index_c, data.edge_weight_c
        if edge_index_c is None:
            raise ValueError("Critic_R: 'edge_index_c' must be provided.")

        h_partition_nodes = self.partition_encoder(x, edge_index_c, edge_weight_c)
        h_partition_agg = self.aggregator(h_partition_nodes, batch)

        raw_params = data.node_params 

        sin_features = torch.sin(raw_params)
        cos_features = torch.cos(raw_params)
        encoded_params = torch.cat([sin_features, cos_features], dim=-1)
        h_q_params_nodes = self.quantum_params_encoder(encoded_params, edge_index_c, edge_weight_c)

        h_q_params_agg = self.aggregator(h_q_params_nodes, batch)

        h_combined = torch.cat([h_graph_agg, h_partition_agg, h_q_params_agg], dim=1)
        ratio = 0.5 * torch.sigmoid(self.prediction_head(h_combined)) + 0.5

        return ratio


        

