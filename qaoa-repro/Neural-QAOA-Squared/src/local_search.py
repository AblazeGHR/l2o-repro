#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.config as config
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected
import numpy as np
import random
import math
import copy
from typing import List, Tuple, Dict
from src.models.critic_r import Critic_R
from src.models.generator import JointGenerator
from torch_geometric.loader import DataLoader as PyGDataLoader
from src.data import ActorGraphDataset


def get_dataloader(graph_data_path: str, batch_size: int) -> PyGDataLoader:

    print(f"Loading preprocessed Actor graph dataset from: {graph_data_path}")

    try:
        dataset = ActorGraphDataset(graph_data_path)
    except FileNotFoundError:
        print(f"[Error] Dataset PKL not found at {graph_data_path}")
        print("Please run 'python data.py --type actor --model train' first.")
        return PyGDataLoader([], batch_size=batch_size) 

    if len(dataset) == 0:
        print("[Warning] Actor graph dataset is empty.")

    num_cpu_cores = os.cpu_count()
    worker_count = min(num_cpu_cores, 8) if num_cpu_cores is not None else 4


    return PyGDataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=worker_count,
        pin_memory=True
    )

def load_partition_generator(model_path: str, device: torch.device, model_filename: str = "generator_best_model_1766545107.pth") -> JointGenerator:
    """
    Args:
        model_path
        device
        model_filename
    
    Returns:
        PartitionGenerator
    """
    if model_filename is None:
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        

        best_model_files = list(model_dir.glob("generator_best_model*.pth"))
        if not best_model_files:
            raise FileNotFoundError(f"No best_model files found in {model_path}")
        
        actual_model_file = max(best_model_files, key=lambda p: p.stat().st_mtime)
        print(f"Auto-selected latest model: {actual_model_file.name}")
    else:
        actual_model_file = os.path.join(model_path, model_filename)
    
    print(f"Loading pre-trained PartitionGenerator model from: {actual_model_file}")
    
    if not os.path.exists(actual_model_file):
        raise FileNotFoundError(f"Model file not found: {actual_model_file}")
    
    model = JointGenerator()
    
    try:
        checkpoint = torch.load(actual_model_file, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"[Error] Failed to load model state_dict from {actual_model_file}.")
        raise
        
    model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
    print("PartitionGenerator model loaded and frozen successfully.")
    return model


def load_critic_r(model_path: str, device: torch.device, model_filename: str ="critic_r_best_model_1766499734.pth") -> Critic_R:
    """
    Args:
        model_path
        Device
        model_filename
    """
    
    if model_filename is None:
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        

        best_model_files = list(model_dir.glob("critic_r_best_model_*.pth"))
        if not best_model_files:
            raise FileNotFoundError(f"No best_model files found in {model_path}")

        actual_model_file = max(best_model_files, key=lambda p: p.stat().st_mtime)
        print(f"Auto-selected latest model: {actual_model_file.name}")
    else:
        actual_model_file = os.path.join(model_path, model_filename)
    
    print(f"Loading pre-trained Critic_R model from: {actual_model_file}")

    if not os.path.exists(actual_model_file):
        print(f"[Error] Critic_R model file not found at: {actual_model_file}")
        raise FileNotFoundError(f"Model file not found: {actual_model_file}")

    critic_r = Critic_R()
    
    try:
        critic_r.load_state_dict(torch.load(actual_model_file, map_location=device)) 
    except Exception as e:
        print(f"[Error] Failed to load model state_dict from {actual_model_file}.") 
        raise
        
    critic_r.to(device)
    critic_r.eval()
    
    for param in critic_r.parameters():
        param.requires_grad = False
        
    print("Critic_R model loaded and frozen successfully.")
    return critic_r

def generate_random_partition(num_nodes: int, max_nodes_per_partition: int) -> List[List[int]]:
    """
    Args:
        num_nodes (int)
        max_nodes_per_partition (int)
    
    Returns:
        List[List[int]]
    """
    # k = ceil(N / max_nodes)
    num_partitions = math.ceil(num_nodes / max_nodes_per_partition)
    
    partitions = [[] for _ in range(num_partitions)]
    partition_sizes = [0] * num_partitions
    
    nodes_shuffled = list(range(num_nodes))
    random.shuffle(nodes_shuffled)
    
    for node in nodes_shuffled:
        available_part_indices = [
            i for i, size in enumerate(partition_sizes) 
            if size < max_nodes_per_partition
        ]
        
        if not available_part_indices:
            raise OverflowError(f"N={num_nodes}, k={num_partitions}, max_nodes={max_nodes_per_partition}")
            
        chosen_part_idx = random.choice(available_part_indices)
        
        partitions[chosen_part_idx].append(node)
        partition_sizes[chosen_part_idx] += 1
        
    return partitions

def convert_partition_to_c_edges(
    partition: List[List[int]], 
    edge_index: torch.Tensor, 
    normalized_edge_attr: torch.Tensor, 
    num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    device = edge_index.device
    
    node_to_part_id = torch.full((num_nodes,), -1, device=device) 
    for i, part in enumerate(partition):
        if part:
            part_tensor = torch.tensor(part, device=device) 
            node_to_part_id[part_tensor] = i


    u_nodes = edge_index[0] # [2*NumEdges]
    v_nodes = edge_index[1] # [2*NumEdges]
    
    part_id_u = node_to_part_id[u_nodes] # [2*NumEdges]
    part_id_v = node_to_part_id[v_nodes] # [2*NumEdges]
    

    keep_mask_c = (part_id_u == part_id_v) & (part_id_u != -1) # [2*NumEdges]
    

    edge_index_c = edge_index[:, keep_mask_c]

    normalized_edge_weight_c = normalized_edge_attr[keep_mask_c].squeeze(-1)


    return edge_index_c, normalized_edge_weight_c

def convert_partition_to_c_edges_from_map(
    node_to_part_id: torch.Tensor, 
    edge_index: torch.Tensor, 
    normalized_edge_attr: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    u_nodes = edge_index[0] # [2*NumEdges]
    v_nodes = edge_index[1] # [2*NumEdges]
    
    part_id_u = node_to_part_id[u_nodes] # [2*NumEdges]
    part_id_v = node_to_part_id[v_nodes] # [2*NumEdges]
    

    keep_mask_c = (part_id_u == part_id_v) & (part_id_u != -1) # [2*NumEdges]
    

    edge_index_c = edge_index[:, keep_mask_c]

    normalized_edge_weight_c = normalized_edge_attr[keep_mask_c].squeeze(-1)
 

    return edge_index_c, normalized_edge_weight_c




def local_search_solver(
    critic_model: Critic_R,
    original_graph_data: Data,
    initial_partition: List[List[int]],
    max_nodes_per_partition: int,
    device: torch.device,
    max_iterations: int = 1000 
) -> Tuple[List[List[int]], float]:

    
    num_nodes = original_graph_data.num_nodes
    num_partitions = len(initial_partition)
    

    node_to_part_id = torch.full((num_nodes,), -1, dtype=torch.long, device=device)

    part_sizes = [0] * num_partitions
    
    for part_id, nodes in enumerate(initial_partition):
        if nodes: 
            part_sizes[part_id] = len(nodes)
            node_to_part_id[torch.tensor(nodes, device=device)] = part_id


    critic_model.eval()
    with torch.no_grad():
        h_graph_nodes_cached = critic_model.topology_encoder(
            original_graph_data.x,
            original_graph_data.edge_index,
            original_graph_data.edge_attr
        ).clone() 

    
    @torch.no_grad()
    def evaluate_from_map(n_to_p_id_tensor: torch.Tensor) -> float:

        edge_index_c, edge_weight_c = convert_partition_to_c_edges_from_map(
            n_to_p_id_tensor,
            original_graph_data.edge_index,
            original_graph_data.edge_attr
        )
        

        eval_data = Data(
            x=original_graph_data.x.clone(), 
            
            edge_index_c=edge_index_c,
            edge_weight_c=edge_weight_c,
            
            h_graph_nodes_cached=h_graph_nodes_cached
            
        )
        
        eval_batch = Batch.from_data_list([eval_data]).to(device)
        
        score = critic_model(eval_batch)
        return score.item()

    current_score = evaluate_from_map(node_to_part_id)

    iteration = 0
    nodes_to_check = list(range(num_nodes))
    
    while iteration < max_iterations:
        iteration += 1
        improved_in_this_loop = False
        random.shuffle(nodes_to_check)

        # --- 1: Move---
        for node_i in nodes_to_check:
            part_i = node_to_part_id[node_i].item() 
            

            for part_k in range(num_partitions):
                if part_i == part_k:
                    continue
                
                if part_sizes[part_k] < max_nodes_per_partition:
                    
                    # 1. Apply - O(1)
                    node_to_part_id[node_i] = part_k
                    part_sizes[part_i] -= 1
                    part_sizes[part_k] += 1
                    
                    # 2. Evaluate
                    neighbor_score = evaluate_from_map(node_to_part_id)
                    
                    # 3. Decide
                    if neighbor_score > current_score:
                        # Accept: keep the change
                        print(f"    [Iter {iteration}, Move] Improvement: {current_score:.6f} -> {neighbor_score:.6f} (Move {node_i} to {part_k})")
                        current_score = neighbor_score
                        improved_in_this_loop = True
                        break # End part_k loop
                    else:
                        # Revert: undo the change - O(1)
                        node_to_part_id[node_i] = part_i
                        part_sizes[part_k] -= 1
                        part_sizes[part_i] += 1
                        
            if improved_in_this_loop:
                break # End node_i loop

        if improved_in_this_loop:
            continue # Restart next major loop (prioritize Move)
        # --- 2: Swap ---
        # (Only executed if no improvement found in Move)
        
        for node_i in nodes_to_check: # Use the same random order
            part_i = node_to_part_id[node_i].item()
            
            # Find node_j in *different* partitions
            for node_j in range(node_i + 1, num_nodes):
                part_j = node_to_part_id[node_j].item()
                
                if part_i != part_j:
                    
                    # 1. Apply swap - O(1)
                    node_to_part_id[node_i] = part_j
                    node_to_part_id[node_j] = part_i
                    
                    # 2. Evaluate
                    neighbor_score = evaluate_from_map(node_to_part_id)
                    
                    # 3. Decide
                    if neighbor_score > current_score:
                        # Accept: keep the change
                        print(f"    [Iter {iteration}, Swap] Improvement: {current_score:.6f} -> {neighbor_score:.6f} (Swap {node_i} & {node_j})")
                        current_score = neighbor_score
                        improved_in_this_loop = True
                        break # End node_j loop
                    else:
                        # Revert: undo the change - O(1)
                        node_to_part_id[node_i] = part_i
                        node_to_part_id[node_j] = part_j
                        
            if improved_in_this_loop:
                break # End node_i loop
                
        if not improved_in_this_loop:
            # Both neighborhoods have been searched without finding improvement
            print(f"  > Reached local optimum at iteration {iteration}.")
            break
    
    if iteration == max_iterations:
        print(f"  > Reached maximum iterations {max_iterations}.")
        
    # --- Step 4: Convert final Tensor back to List[List] for return ---
    final_partition_list = [[] for _ in range(num_partitions)]
    for node_idx, part_id in enumerate(node_to_part_id.cpu().tolist()):
        if part_id != -1: # Ensure node is assigned
            final_partition_list[part_id].append(node_idx)
            
    return final_partition_list, current_score


def simulated_annealing_solver(
    critic_model: Critic_R,
    original_graph_data: Data,
    initial_partition: List[List[int]],
    max_nodes_per_partition: int,
    device: torch.device,
    # --- Introduce SA and N-tournament hyperparameters ---
    tournament_size: int,
    max_iterations: int,
    initial_temperature: float,
    cooling_rate: float,
    min_temperature: float,
    # --- [Optimization 3] Introduce adjustable move/swap ratio ---
    move_swap_ratio: float = 0.8 # 80% Move, 20% Swap (usually Move is more important)
) -> Tuple[List[List[int]], float]:

    
    num_nodes = original_graph_data.num_nodes
    num_partitions = len(initial_partition)
    
    # --- Step 1: [Optimization 2 & 3] 
    # More efficiently convert List[List] to Tensor (node_to_part_id)
    node_to_part_id = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    part_sizes_tensor = torch.zeros(num_partitions, dtype=torch.long, device=device)
    
    for part_id, nodes in enumerate(initial_partition):
        if nodes: 
            # Only create tensor for non-empty partitions
            nodes_tensor = torch.tensor(nodes, dtype=torch.long, device=device)
            part_sizes_tensor[part_id] = len(nodes)
            node_to_part_id[nodes_tensor] = part_id

    # --- Step 1.5: Precompute and cache GNN topology encoding (done) ---
    print("  > Precomputing GNN topology encoding (h_graph_nodes_cached)...")
    critic_model.eval() 
    with torch.no_grad():
        # This leverages the logic in critic_r.py: "if hasattr(data, 'h_graph_nodes_cached')"
        h_graph_nodes_cached = critic_model.topology_encoder(
            original_graph_data.x,
            original_graph_data.edge_index,
            original_graph_data.edge_attr
        ).clone()
        
        # --- [Optimization 1.1] Precompute static h_graph_pooled and batch_vector ---
        # Because we only handle a single graph, batch_vector is always all zeros
        batch_vector = torch.zeros(num_nodes, dtype=torch.long, device=device)
        
        # h_graph_nodes_cached never changes, so h_graph_pooled_cached never changes
        # This corresponds to critic_r.py: self.aggregator(h_graph_nodes, batch)
        h_graph_pooled_cached = global_mean_pool(
            h_graph_nodes_cached, batch_vector
        )
    print("  > Topology encoding and static pooling cache completed.")

    # --- Step 2: [Optimization 1.2] Define *super efficient* internal evaluation function ---
    @torch.no_grad()
    def evaluate_from_map(n_to_p_id_tensor: torch.Tensor) -> float:
        """ 
        [Internal] [Optimized]
        Bypass Data/Batch creation, directly call GNN submodules
        (This function manually replicates the forward logic of critic_r.py)
        """
        # 1. Compute dynamic c_edges (this is still necessary)
        edge_index_c, edge_weight_c = convert_partition_to_c_edges_from_map(
            n_to_p_id_tensor,
            original_graph_data.edge_index,
            original_graph_data.edge_attr
        )
        
        # 2. [Core] Manually execute the dynamic part of Critic_R.forward
        
        # Step 2.1: Run partition_encoder
        # Corresponds to critic_r.py: h_partition_nodes = self.partition_encoder(x, edge_index_c, ...)
        h_part = critic_model.partition_encoder(
            original_graph_data.x, 
            edge_index_c, 
            edge_weight_c
        )
        
        # Step 2.2: Run global_mean_pool on h_part
        # Corresponds to critic_r.py: h_partition_agg = self.aggregator(h_partition_nodes, batch)
        h_part_pooled = global_mean_pool(h_part, batch_vector) 
        
        # Step 2.3: Combine
        # Corresponds to critic_r.py: h_combined = torch.cat([h_graph_agg, h_partition_agg], dim=1)
        combined_h = torch.cat([h_graph_pooled_cached, h_part_pooled], dim=1)
        
        # Step 2.4: Run predictor
        # Corresponds to critic_r.py: return 0.5 * torch.sigmoid(self.prediction_head(h_combined)) + 0.5
        score = critic_model.prediction_head(combined_h)
        
        # Return the exact same scaling as in critic_r.py
        return (0.5 * torch.sigmoid(score) + 0.5).item()

    # --- Step 3: Execute (N-tournament + SA) search ---
    T = initial_temperature
    current_score = evaluate_from_map(node_to_part_id)
    print(f"  > Initial score: {current_score:.6f}")
    
    best_score = current_score
    best_partition_map = node_to_part_id.clone()
        
    for iteration in range(max_iterations):
        if T < min_temperature:
            print(f"  > [Iter {iteration}] Temperature ({T:.2e}) below minimum. Stopping.")
            break

        # --- Step 3.1: N-tournament ---
        best_neighbor_score = -float('inf')
        best_neighbor_move = None 
        
        for _ in range(tournament_size):
            neighbor_score = -1.0
            move_details = None
            
            # [Optimization 4] Use configurable move_swap_ratio (yours is 0.5, I changed it back to 0.8)
            if random.random() < move_swap_ratio:
                # --- Neighborhood 1: "Move" ---
                node_i = random.randrange(num_nodes)
                part_i = node_to_part_id[node_i].item()
                part_k = random.randrange(num_partitions)
                
                # [Optimization 2] Use part_sizes_tensor
                if part_i == part_k or part_sizes_tensor[part_k] >= max_nodes_per_partition:
                    continue 
                
                # 1. Apply
                node_to_part_id[node_i] = part_k
                part_sizes_tensor[part_i] -= 1
                part_sizes_tensor[part_k] += 1
                
                # 2. Evaluate [Now very fast]
                neighbor_score = evaluate_from_map(node_to_part_id)
                move_details = ("move", node_i, part_k, part_i) 
                
                # 3. Revert
                node_to_part_id[node_i] = part_i
                part_sizes_tensor[part_k] -= 1
                part_sizes_tensor[part_i] += 1
                
            else:
                # --- Neighborhood 2: "Swap" ---
                node_i = random.randrange(num_nodes)
                node_j = random.randrange(num_nodes)
                part_i = node_to_part_id[node_i].item()
                part_j = node_to_part_id[node_j].item()
                
                if node_i == node_j or part_i == part_j:
                    continue

                # 1. Apply
                node_to_part_id[node_i] = part_j
                node_to_part_id[node_j] = part_i
                
                # 2. Evaluate [Now very fast]
                neighbor_score = evaluate_from_map(node_to_part_id)
                move_details = ("swap", node_i, node_j, part_i, part_j)
                
                # 3. Revert
                node_to_part_id[node_i] = part_i
                node_to_part_id[node_j] = part_j

            if neighbor_score > best_neighbor_score:
                best_neighbor_score = neighbor_score
                best_neighbor_move = move_details

        # --- Step 3.2: SA decision ---
        if best_neighbor_move is None:
            T *= cooling_rate 
            continue

        delta_score = best_neighbor_score - current_score
        
        if delta_score > 0 or random.random() < math.exp(delta_score / T):
            # Accept: permanently apply the tournament champion move
            current_score = best_neighbor_score
            move_info_str = ""
            
            if best_neighbor_move[0] == "move":
                m_type, node_i, part_k, part_i = best_neighbor_move
                node_to_part_id[node_i] = part_k
                part_sizes_tensor[part_i] -= 1 # [Optimization 2]
                part_sizes_tensor[part_k] += 1 # [Optimization 2]
                move_info_str = f"Move {node_i}: p{part_i} -> p{part_k}"
            
            elif best_neighbor_move[0] == "swap":
                m_type, node_i, node_j, part_i, part_j = best_neighbor_move
                node_to_part_id[node_i] = part_j
                node_to_part_id[node_j] = part_i
                # (part_sizes_tensor does not change in swap)
                move_info_str = f"Swap {node_i}(p{part_i}) <-> {node_j}(p{part_j})"

            if current_score > best_score:
                best_score = current_score
                best_partition_map = node_to_part_id.clone()
                print(f"    [Iter {iteration}, T={T:.2e}] New best: {best_score:.6f} ({move_info_str})")
            
        # --- Step 3.3: Cooling ---
        T *= cooling_rate
        
    if iteration == max_iterations - 1:
        print(f"  > Reached maximum iterations {max_iterations}.")
        
    # --- Step 4: Convert the *best* Tensor back to List[List] ---
    final_partition_list = [[] for _ in range(num_partitions)]
    for node_idx, part_id in enumerate(best_partition_map.cpu().tolist()):
        if part_id != -1: 
            final_partition_list[part_id].append(node_idx)
            
    return final_partition_list, best_score


# --- 4. Main execution program ---
def main():
    """
    Main function to execute the local search process
    """
    
    # --- 0. Configuration ---
    BATCH_SIZE = config.LOCAL_SEARCH_BATCH_SIZE             # Number of initial random partitions
    MAX_NODES_PER_PARTITION = config.MAX_NODES_PER_PARTITION  # Maximum number of qubits per partition
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Prepare model and data ---
    
    # (a) Load the pre-trained Critic_R model
    print("\n--- Preparing pre-trained Critic_R model ---")
    critic_r = load_critic_r(config.CRITIC_MODEL_PATH, device)
    
    # (b) Prepare the original graph data G

    dataloader = get_dataloader(config.ACTOR_GRAPH_DATASET_PATH, batch_size=1)
    # <--- Modification: Load only the first data point (batch) ---
    print("Loading *only the first batch* for local search...")
    try:
        batch_data = next(iter(dataloader))
    except StopIteration:
        print("[Error] Dataloader is empty. Cannot train.")
        return

    # PyG DataLoader returns a Batch (even if batch_size=1),
    # local_search needs a single Data object -> convert to Data and move to device
    if hasattr(batch_data, "to_data_list"):
        data0 = batch_data.to_data_list()[0]
    else:
        data0 = batch_data

    # Move Data to device
    data0 = data0.to(device)
    graph_data_instance = data0
    NUM_NODES = graph_data_instance.num_nodes
    # ---------------------------------------------

    # --- 2. Generate BATCH_SIZE initial solutions ---
    
    print(f"\n--- Step 1: Generate {BATCH_SIZE} random initial partitions ---")
    initial_partitions = []
    for i in range(BATCH_SIZE):
        pi_0 = generate_random_partition(NUM_NODES, MAX_NODES_PER_PARTITION)
        initial_partitions.append(pi_0)
        
    print(f"Generated {len(initial_partitions)} initial solutions.")
    print(f"Each solution has k={len(initial_partitions[0])} partitions (based on N={NUM_NODES}, max_nodes={MAX_NODES_PER_PARTITION})")

    # --- 3. Run local search in parallel ---
    
    print(f"\n--- Step 2: Run local search on {BATCH_SIZE} initial solutions ---")
    
    best_overall_score = -1.0  # Critic_R outputs in [0.5, 1.0], -1.0 is a safe initial value
    best_overall_partition = None
    
    # Note: This is a serial loop.
    # To achieve true parallelism, you can use `torch.multiprocessing` or `joblib`
    # Run local_search_solver in parallel on BATCH_SIZE CPU cores
    
    for i, pi_0 in enumerate(initial_partitions):
        print(f"\n--- [ Run {i+1} / {BATCH_SIZE} ] ---")
        
        final_partition, final_score = local_search_solver(
            critic_r,
            graph_data_instance,
            pi_0,
            MAX_NODES_PER_PARTITION,
            device
        )
        
        print(f"--- [ Run {i+1} completed ] Final score: {final_score:.12f} ---")
        
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall_partition = final_partition
            print(f"!!! Found a new global best solution !!!")

    # --- 4. Output final results ---
    
    print("\n\n--- Local search completed ---")
    print(f"Best partition (pi^*) predicted score: {best_overall_score:.12f}")
    
    # (Optional) Print details of the best partition
    print("Best partition details:")
    for part_id, nodes in enumerate(best_overall_partition):
        print(f"  Partition {part_id} ({len(nodes)} nodes): {nodes}")

if __name__ == "__main__":
    main()