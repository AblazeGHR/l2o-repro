import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.config
from pennylane import numpy as np
import numpy
import math
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import random
import igraph as ig
import itertools
import heapq
import signal
import pymetis
import torch
from torch_geometric.data import Data, Batch
from src.utils import calculate_node_features
from src.local_search import (
    load_critic_r,
    load_partition_generator,
    local_search_solver,
    simulated_annealing_solver,
    generate_random_partition,
)
import os
from pathlib import Path
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm


class Graph():
    '''
    A graph is saved in both an adjoint matrix and edge list.
    '''
    def __init__(self, v: list = None, edges: list = None, adjoint=None) -> None:
        self.v = [int(node) for node in v]
        self.n_v = len(self.v)

        if edges is not None:
            self.e = [(int(u), int(v), float(w)) for u, v, w in edges]
        else:
            self.e = None

        self.adj = adjoint

        if self.adj is None:
            self._edges_to_adjoint()

        if self.e is None:
            self._adjoint_to_edges()

        self.v2i = {v[i]: i for i in range(self.n_v)}

    def _edges_to_adjoint(self) -> None:
        self.adj = np.zeros((self.n_v, self.n_v), requires_grad=False)
        for edge in self.e:
            v1 = edge[0]
            v2 = edge[1]
            w = edge[2]
            self.adj[v1][v2] = w
            self.adj[v2][v1] = w

    def _adjoint_to_edges(self) -> None:
        self.e = []

        for i in range(self.n_v):
            for j in range(i + 1, self.n_v):
                if self.adj[i][j] != 0:
                    self.e.append((i, j, self.adj[i][j].item()))

    def _calculate_boundary_nodes(self, partitions: list) -> int:
        """Helper function to calculate the total number of boundary nodes."""
        boundary_nodes = set()
        node_to_partition_map = {}
        for i, part in enumerate(partitions):
            for node in part:
                node_to_partition_map[node] = i

        for i, part in enumerate(partitions):
            for node in part:
                row_data = numpy.array(self.adj[node])
                neighbors = numpy.nonzero(row_data)[0]
                for neighbor in neighbors:
                    idx = int(neighbor)
                    part = node_to_partition_map.get(idx)
                    if part is not None and part != i:
                        boundary_nodes.add(node)
                        break
        return len(boundary_nodes)

    def _is_boundary(self, node: int, node_to_part_idx: dict) -> bool:
        """
        Efficiently checks if a single node is a boundary node.
        A node is a boundary node if it has at least one neighbor in a different partition.
        """
        if node not in self.v2i:
            return False
        row_data = numpy.array(self.adj[node])
        neighbors = numpy.nonzero(row_data)[0]
        if len(neighbors) == 0:
            return False

        source_idx = node_to_part_idx[node]
        for neighbor in neighbors:
            if neighbor in node_to_part_idx and node_to_part_idx[neighbor] != source_idx:
                return True
        return False

    def graph_partition(self, n: int, policy: str = 'random', n_sub=1, depth: int = 1) -> list:
        '''
        n : Allowable qubit number.

        policy : Partition strategy. Default is 'random'. Options include:
            'random' : Random Partition.
            'modularity' : Greedy Modularity Maximization.
            'boundary' : Boundary Vertices Minimization.
            'kl' : Kernighan-Lin Algorithm.
            'JointGenerator+Critic' : Learnable Partitioning with pre-trained models.
        n_sub : number of subgraphs.
        depth : depth of QAOA.
        ''' 
        H = []
        v = self.v
        init_gammas_betas = None

        if policy == 'JointGenerator+Critic':
            G_nx = nx.Graph()
            sorted_nodes = sorted(self.v)
            G_nx.add_nodes_from(sorted_nodes)
            for edge in self.e:
                u, v, w = edge[0], edge[1], edge[2]
                G_nx.add_edge(u, v, weight=w)

            features = calculate_node_features(G_nx)

            node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}

            if self.e:
                edge_sources = [node_to_idx[e[0]] for e in self.e]
                edge_targets = [node_to_idx[e[1]] for e in self.e]
                edge_weights = [e[2] for e in self.e]

                _edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
                _edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

                edge_index = torch.cat([_edge_index, _edge_index.flip(0)], dim=1)
                edge_attr = torch.cat([_edge_attr, _edge_attr], dim=0)

                if edge_attr.numel() > 0:
                    max_abs_val = torch.abs(edge_attr).max()
                    if max_abs_val > 1e-9:
                        normalized_edge_attr = edge_attr / max_abs_val
                    else:
                        normalized_edge_attr = edge_attr
                else:
                    normalized_edge_attr = edge_attr
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                normalized_edge_attr = torch.empty((0, 1), dtype=torch.float32)

            graph_data = Data(
                x=features,
                edge_index=edge_index,
                edge_attr=normalized_edge_attr,
            )
            graph_data.batch = torch.zeros(self.n_v, dtype=torch.long)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            partition_generator_model = load_partition_generator(
                src.config.PARTITION_GENERATOR_MODEL_PATH, device
            )
            critic_model = load_critic_r(src.config.CRITIC_MODEL_PATH, device)

            partition_generator_model.train()

            for param in partition_generator_model.parameters():
                param.requires_grad = True

            print("Optimizing PartitionGenerator model on the fly...")

            OPTIMIZE_STEPS = 64
            # OPTIMIZE_STEPS = 1, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097 # TTA analysis
            optimizer = optim.AdamW(
                partition_generator_model.parameters(),
                lr=src.config.INFERENCE_LEARNING_RATE,
                weight_decay=src.config.GENERATOR_WEIGHT_DECAY,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.8,
                patience=100,
                min_lr=1e-4,
            )

            k = math.ceil(1.0 * self.n_v / n)

            graph_data = graph_data.to(device)

            best_l_total_for_this_graph = float('inf')
            best_partition_indices = None
            best_P = None
            initial_l_total = 0.0
            min_lr = 1e-4

            if OPTIMIZE_STEPS == 0:
                partition_generator_model.eval()
                with torch.no_grad():
                    batch_data = graph_data.clone()
                    p_indices, P_val, data_out = partition_generator_model(
                        batch_data, k, src.config.MAX_NODES_PER_PARTITION
                    )

                    score = critic_model(data_out)
                    if score.dim() > 1:
                        score = score.view(-1)
                    loss = -score

                    current_loss = loss.item()
                    initial_l_total = current_loss
                    best_l_total_for_this_graph = current_loss
                    best_partition_indices = p_indices.detach().clone()
                    best_P = P_val.detach().clone()

                    print(f"Inference Only (Step 0): Loss={current_loss:.6f}")

            elif OPTIMIZE_STEPS > 0:
                partition_generator_model.train()

                inner_pbar = tqdm(
                    range(OPTIMIZE_STEPS),
                    desc=f"Optimizing Graph with {self.n_v} nodes into {k} partitions",
                    leave=False,
                )

                for step in inner_pbar:
                    batch_data = graph_data.clone()

                    optimizer.zero_grad()

                    partition_indices, P, batch_data = partition_generator_model(
                        batch_data,
                        k,
                        src.config.MAX_NODES_PER_PARTITION,
                    )

                    l_total = -critic_model(batch_data)

                    current_loss = l_total.item()

                    if step == 0:
                        initial_l_total = current_loss

                    if current_loss < best_l_total_for_this_graph:
                        best_l_total_for_this_graph = current_loss
                        best_partition_indices = partition_indices.detach().clone()
                        best_P = P.detach().clone()

                    if step < OPTIMIZE_STEPS - 1:
                        l_total.backward()
                        optimizer.step()
                        scheduler.step(current_loss)

                    inner_pbar.set_postfix(
                        Loss=f"{current_loss:.6f}",
                        Best=f"{best_l_total_for_this_graph:.6f}",
                        LR=f"{optimizer.param_groups[0]['lr']:.1e}",
                    )

                    if abs(optimizer.param_groups[0]['lr'] - min_lr) < 1e-8:
                        break

            print(
                f"Initial L_Total: {initial_l_total:.6f}, Best L_Total: {best_l_total_for_this_graph:.6f}"
            )

            partition_list = [[] for _ in range(k)]
            for node_idx, part_idx in enumerate(best_partition_indices.detach().cpu().tolist()):
                partition_list[part_idx].append(node_idx)

            best_P_numpy = best_P.detach().cpu().numpy()
            best_P_all = best_P_numpy.reshape(-1, 2, src.config.QAOA_DEPTH)

            init_gammas_betas_list = []

            for part_idx, node_indices in enumerate(partition_list):
                if node_indices:
                    actual_nodes = [sorted_nodes[idx] for idx in node_indices]
                    A = self.adj[actual_nodes][:, actual_nodes]
                    H.append(Graph(v=actual_nodes, adjoint=A))

                    current_params = best_P_all[part_idx]
                    init_gammas_betas_list.append(current_params)

            init_gammas_betas = numpy.array(init_gammas_betas_list)

        if policy == 'random':
            n_sub = math.ceil(self.n_v / n)
            v_copy = list(self.v)
            np.random.shuffle(v_copy)

            sub_arrays = np.array_split(v_copy, n_sub)
            sub_list = [list(arr) for arr in sub_arrays]

            for i in range(n_sub):
                A = self.adj[sub_list[i]][:, sub_list[i]]
                H.append(Graph(v=sub_list[i], adjoint=A))

        if policy == 'modularity':
            G = nx.Graph()
            G.add_nodes_from(v)
            for x in self.e:
                G.add_edge(x[0], x[1], weight=x[2])
            c = greedy_modularity_communities(G)
            initial_partitions = []
            sub_list = [list(x) for x in c]
            for x in sub_list:
                if len(x) > n:
                    np.random.shuffle(x)
                    n_ssub = math.ceil(len(x) / n)

                    ssub_list = [x[n * i : n * (i + 1)] for i in range(n_ssub)]
                    initial_partitions.extend(ssub_list)
                else:
                    initial_partitions.append(x)

            while True:
                best_merge_candidates = []
                max_connection_strength = -1
                for i in range(len(initial_partitions)):
                    for j in range(i + 1, len(initial_partitions)):
                        p1 = initial_partitions[i]
                        p2 = initial_partitions[j]
                        if len(p1) + len(p2) > n:
                            continue
                        connection_strength = np.sum(self.adj[p1, :][:, p2])
                        if connection_strength > max_connection_strength:
                            max_connection_strength = connection_strength
                            best_merge_candidates = [(i, j)]
                        elif connection_strength == max_connection_strength:
                            best_merge_candidates.append((i, j))

                if best_merge_candidates and max_connection_strength >= 0:
                    i, j = random.choice(best_merge_candidates)

                    if i > j:
                        i, j = j, i

                    merged_partition = initial_partitions[i] + initial_partitions[j]

                    initial_partitions.pop(j)
                    initial_partitions.pop(i)
                    initial_partitions.append(merged_partition)
                else:
                    break

            for nodes in initial_partitions:
                A = self.adj[nodes][:, nodes]
                H.append(Graph(v=nodes, adjoint=A))

        if policy == 'boundary':
            G_nx = nx.Graph()
            G_nx.add_nodes_from(v)
            for x in self.e:
                G_nx.add_edge(x[0], x[1], weight=x[2])

            total_weight = G_nx.size(weight='weight')
            if G_nx.number_of_edges() == 0 or total_weight == 0:
                if G_nx.number_of_nodes() > 0:
                    partitions = [[n for n in G_nx.nodes()]]
                else:
                    partitions = []
            else:
                communities = nx.algorithms.community.louvain_communities(G_nx, weight='weight')
                partitions = [list(p) for p in communities]

            print(f"DEBUG: Graph generated {len(partitions)} initial partitions.")

            round_counter = 0

            while True:
                round_counter += 1
                improved_in_this_round = False

                current_boundary_total = self._calculate_boundary_nodes(partitions)

                node_to_part_idx = {
                    node: i for i, part in enumerate(partitions) for node in part
                }

                nodes_list = list(self.v)
                numpy.random.shuffle(nodes_list)

                for node in nodes_list:
                    source_idx = node_to_part_idx[node]

                    neighbors = numpy.nonzero(numpy.array(self.adj[node]))[0]

                    candidate_partitions = set()
                    for nb in neighbors:
                        nb_idx = int(nb)
                        if nb_idx in node_to_part_idx:
                            candidate_partitions.add(node_to_part_idx[nb_idx])

                    if source_idx in candidate_partitions:
                        candidate_partitions.remove(source_idx)

                    if not candidate_partitions:
                        continue

                    for target_idx in candidate_partitions:
                        if len(partitions[target_idx]) + 1 > n:
                            continue

                        partitions[source_idx].remove(node)
                        partitions[target_idx].append(node)

                        new_boundary_total = self._calculate_boundary_nodes(partitions)

                        if new_boundary_total < current_boundary_total:
                            current_boundary_total = new_boundary_total
                            node_to_part_idx[node] = target_idx

                            improved_in_this_round = True
                            break
                        else:
                            partitions[target_idx].remove(node)
                            partitions[source_idx].append(node)

                partitions = [p for p in partitions if p]

                if not improved_in_this_round:
                    print(f"[Stage 2] Converged after {round_counter} rounds.")
                    break

            final_partitions = partitions

            processed_partitions = []
            for part in final_partitions:
                if len(part) > n:
                    np.random.shuffle(part)
                    num_sub_parts = math.ceil(len(part) / n)
                    for i in range(num_sub_parts):
                        processed_partitions.append(part[i * n : (i + 1) * n])
                else:
                    processed_partitions.append(part)

            while True:
                best_merge_candidates = []
                max_connection_strength = -1
                for i in range(len(processed_partitions)):
                    for j in range(i + 1, len(processed_partitions)):
                        p1 = processed_partitions[i]
                        p2 = processed_partitions[j]
                        if len(p1) + len(p2) > n:
                            continue
                        connection_strength = np.sum(self.adj[p1, :][:, p2])
                        if connection_strength > max_connection_strength:
                            max_connection_strength = connection_strength
                            best_merge_candidates = [(i, j)]
                        elif connection_strength == max_connection_strength:
                            best_merge_candidates.append((i, j))

                if best_merge_candidates and max_connection_strength >= 0:
                    i, j = random.choice(best_merge_candidates)
                    if i > j:
                        i, j = j, i
                    merged = processed_partitions[i] + processed_partitions[j]
                    processed_partitions.pop(j)
                    processed_partitions.pop(i)
                    processed_partitions.append(merged)
                else:
                    break

            for nodes in processed_partitions:
                A = self.adj[nodes][:, nodes]
                H.append(Graph(v=nodes, adjoint=A))

        if policy == 'kl':
            final_partitions = []

            def recursive_bisection(nodes_to_split):
                if len(nodes_to_split) <= n:
                    final_partitions.append(nodes_to_split)
                    return

                subgraph_nx = nx.Graph()
                subgraph_nx.add_nodes_from(nodes_to_split)
                adj_slice = self.adj[nodes_to_split, :][:, nodes_to_split]
                rows, cols = np.nonzero(adj_slice)
                for i, j in zip(rows, cols):
                    if i < j:
                        subgraph_nx.add_edge(
                            nodes_to_split[i],
                            nodes_to_split[j],
                            weight=abs(adj_slice[i, j]),
                        )

                if subgraph_nx.number_of_edges() == 0:
                    final_partitions.append(nodes_to_split[:n])
                    if len(nodes_to_split) > n:
                        recursive_bisection(nodes_to_split[n:])
                    return

                part1, part2 = nx.algorithms.community.kernighan_lin_bisection(
                    subgraph_nx, weight='weight'
                )

                recursive_bisection(list(part1))
                recursive_bisection(list(part2))

            if self.v:
                recursive_bisection(list(self.v))

            for nodes in final_partitions:
                if not nodes:
                    continue
                A = self.adj[nodes][:, nodes]
                H.append(Graph(v=nodes, adjoint=A))

        if init_gammas_betas is None:
            init_gammas_betas = np.random.uniform(
                low=0.0,
                high=2 * np.pi,
                size=(len(H), 2, depth),
            )

        return H, init_gammas_betas
