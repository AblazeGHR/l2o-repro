# utils.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import pickle
import torch
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data
from collections import defaultdict
from typing import List, Tuple



# [TEST PASS]
def format_bytes(bytes_val):
    """Format bytes as MB or GB"""
    if bytes_val >= 1024**3:
        return f"{bytes_val / 1024**3:.2f} GB"
    return f"{bytes_val / 1024**2:.1f} MB"

# [TEST PASS]
def calculate_sum_neg_weights(file_path):
    """
    Read a graph file and calculate the sum of all negative edge weights (SumNeg).
    
    Args:
        file_path (Path): Path to the .txt graph file.

    Returns:
        float: Sum of all negative weights, or None if the file cannot be processed.
    """
    sum_neg = 0.0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Skip the first line header
            for line in lines[1:]:
                if line.strip():
                    parts = line.strip().split()
                    # Weight is the third element
                    if len(parts) >= 3:
                        weight = float(parts[2])
                        if weight < 0:
                            sum_neg += weight
    except (IOError, ValueError, IndexError) as e:
        print(f"Error reading or parsing graph file {file_path} for SumNeg calc: {e}")
        return None  # Return None on error
    return sum_neg

# [TEST PASS]
def parse_txt_to_nx_graph(file_path: Path) -> nx.Graph:
    """
    Parse a Biq Mac .txt file and construct a NetworkX graph.
    
    Assumes the .txt file format is as follows:
    First line: num_nodes num_edges
    Subsequent lines: node1 node2 weight (assumes nodes are 1-indexed)
    """
    g = nx.Graph()
    num_nodes = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            if lines:
                header = lines[0].strip().split()
                if len(header) >= 1:
                    num_nodes = int(header[0])
                    g.add_nodes_from(range(num_nodes)) 
            
            for line in lines[1:]:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        # Assume the file is 1-indexed, convert to 0-indexed
                        u, v, w = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
                        g.add_edge(u, v, weight=w)
                        
    except (IOError, ValueError, IndexError) as e:
        print(f"Error parsing graph file {file_path}: {e}")
        return None
                
    return g

# [TEST PASS]
def parse_graph_from_pyg_json(graph_json: dict) -> nx.Graph:
    """Parse a NetworkX graph from a torch_geometric-style JSON object"""
    g = nx.Graph()
    if 'nodes' in graph_json:
        g.add_nodes_from(graph_json['nodes'])
    
    edge_index = graph_json['edge_index']
    edge_weights = graph_json.get('edge_weight', [1.0] * len(edge_index[0]))

    for i in range(len(edge_index[0])):
        u, v, w = int(edge_index[0][i]), int(edge_index[1][i]), float(edge_weights[i])
        g.add_edge(u, v, weight=w)
        
    return g


def partition_to_edge_index_and_weight_c(
    partition: List[List[int]], 
    original_graph: nx.Graph
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a partitioning scheme into a new graph's edge index (edge_index_c) and edge weights (edge_weight_c).
    
    This new graph only includes edges from the original graph whose both endpoints lie within the same partition.
    Weights are extracted from the original graph's 'weight' attribute.
    """
    if not partition:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
        
    node_to_partition_map = {node: i for i, part in enumerate(partition) for node in part}
    
    intra_partition_edges = []
    intra_partition_weights = []

    warning_printed = False
    
    # Use data=True to get edge attributes
    for u, v, data in original_graph.edges(data=True):
        part_u = node_to_partition_map.get(u)
        part_v = node_to_partition_map.get(v)
        
        # Core logic: check if both endpoints are in the same partition
        if part_u is not None and part_u == part_v:
            # If in the same partition, keep this original edge (u, v)
            intra_partition_edges.append((u, v))
            
            # Get weight. If 'weight' attribute is missing, default to 1.0
            if 'weight' in data:
                weight = data['weight']
            else:
                weight = 1.0
                # Only print warning once when first encountered
                if not warning_printed:
                    # You can also use warnings.warn("...")
                    print("Warning: Edge without 'weight' attribute found. Defaulting to weight=1.0.")
                    warning_printed = True
            intra_partition_weights.append(weight)
            
    if not intra_partition_edges:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
    
    # --- Convert to PyG format ---
    
    # 1. Convert edge index
    edge_index_c = torch.tensor(intra_partition_edges, dtype=torch.long).t().contiguous()
    
    # 2. Convert edge weights
    # Note: weights are typically floating-point numbers
    edge_weight_c = torch.tensor(intra_partition_weights, dtype=torch.float)
    
    # 3. Add reverse edges for undirected graph

    # 4. Duplicate weights for reverse edges
    # The weight of (u, v) is the same as (v, u)

    return torch.cat([edge_index_c, edge_index_c.flip(0)], dim=1), torch.cat([edge_weight_c, edge_weight_c], dim=0)

# [TEST PASS]
def partition_to_edge_index_c(
    partition: List[List[int]],
    original_graph: nx.Graph
) -> torch.Tensor:
    """
    Convert a partitioning scheme into a new graph's edge index (edge_index_c).
    This new graph only includes edges from the original graph whose both endpoints lie within the same partition.
    """
    if not partition:
        return torch.empty((2, 0), dtype=torch.long)
        
    node_to_partition_map = {node: i for i, part in enumerate(partition) for node in part}
    
    intra_partition_edges = []
    # Iterate over each edge in the original graph
    for u, v in original_graph.edges():
        part_u = node_to_partition_map.get(u)
        part_v = node_to_partition_map.get(v)
        
        # Core logic: check if both endpoints are in the same partition
        if part_u is not None and part_u == part_v:
            # If in the same partition, keep this original edge (u, v)
            intra_partition_edges.append((u, v))
            
    if not intra_partition_edges:
        return torch.empty((2, 0), dtype=torch.long)
    
    # Convert to PyG's edge_index format
    edge_index_c = torch.tensor(intra_partition_edges, dtype=torch.long).t().contiguous()
    
    # Add reverse edges for undirected graph
    return torch.cat([edge_index_c, edge_index_c.flip(0)], dim=1)

# ===============================================================
# 3. Node Feature Calculation Functions
# ===============================================================

# [TEST PASS]
def calculate_node_features(g: nx.Graph):
    """
    Calculate and return the node feature matrix for graph g.

    Calculated features include:
    1.  Node Degree: The number of neighbors a node has, a fundamental local importance metric.
    2.  Weighted Degree: If the graph has weights, this is a more meaningful measure of connection strength.
    3.  Clustering Coefficient: Measures how tightly connected a node's neighbors are, reflecting community structure.
    4.  PageRank: Measures a node's global "influence" in the network.
    5.  Betweenness: Measures a node's ability to act as a "bridge" in the network.
    Args:
        g (nx.Graph): Input NetworkX graph.

    Returns:
        torch.Tensor: A PyTorch tensor of shape [N, F],
                      where N is the number of nodes and F is the feature dimension (5 in this case).
    
    """
    # Ensure consistent node order; NetworkX usually maintains order, but explicit specification is safer
    nodes_order = sorted(list(g.nodes()))
    
    abs_weights = {edge: abs(g.edges[edge]['weight']) for edge in g.edges()}
    nx.set_edge_attributes(g, abs_weights, 'abs_weight')

    # --- Tier 1: Basic Structural Features ---
    
    # 1. Node Degree (unweighted)
    degrees = np.array([g.degree[n] for n in nodes_order])

    # 2. Weighted Node Degree
    weighted_degrees_dict = dict(g.degree(weight='weight'))
    weighted_degrees = np.array([weighted_degrees_dict.get(n, 0) for n in nodes_order])

    # 3. Clustering Coefficient
    # Calculated using 'abs_weight' for more meaningful local connectivity measure
    clustering_coeffs_dict = nx.clustering(g, weight='abs_weight')
    clustering_coeffs = np.array([clustering_coeffs_dict.get(n, 0) for n in nodes_order])

    # --- Tier 2 Features ---

    # 4. PageRank
    # Calculated using 'abs_weight' for edge weights
    try:
        pagerank_dict = nx.pagerank(g, weight='abs_weight')
    except nx.PowerIterationFailedConvergence:
        print(f"Warning: PageRank failed to converge for graph. Using uniform distribution as fallback.")
        pagerank_dict = {n: 1.0 / g.number_of_nodes() for n in nodes_order}
    pageranks = np.array([pagerank_dict.get(n, 0) for n in nodes_order])

    # 5. Betweenness Centrality (computationally expensive)
    # Calculated using 'abs_weight'
    betweenness_dict = nx.betweenness_centrality(g, weight='abs_weight')
    betweenness = np.array([betweenness_dict.get(n, 0) for n in nodes_order])

    for u, v in g.edges():
        if 'abs_weight' in g[u][v]: del g[u][v]['abs_weight']

    # --- Integration and Standardization ---

    feature_list = [degrees, weighted_degrees, clustering_coeffs, pageranks, betweenness]
    
    standardized_features = []
    for feature in feature_list:
        mean = feature.mean()
        std = feature.std()
        standardized_feature = (feature - mean) / (std + 1e-6)
        standardized_features.append(standardized_feature)
        
    features_matrix = np.stack(standardized_features, axis=1)
    
    return torch.tensor(features_matrix, dtype=torch.float32)