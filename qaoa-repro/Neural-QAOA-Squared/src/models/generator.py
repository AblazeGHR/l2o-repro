# generator.py
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
from .partition_generator import PartitionGenerator
from .param_generator import ParamGenerator


class JointGenerator(nn.Module):
    """
    Joint generator (JointGenerator): includes partition generation (Partition) and parameter generation (Parameter).
    The architecture follows a two-stage design: "Partition-First, Parameter-Second".
    """
    def __init__(self):
        super(JointGenerator, self).__init__()
        
        # 1. Initialize partition generator (Partition Actor)
        # Responsible for generating the node partition S for the input graph
        # Architecture: GAT -> OCH -> GCD
        self.partition_gen = PartitionGenerator()

        # 2. Initialize parameter generator (Parameter Actor)
        # Responsible for generating QAOA parameters P conditioned on partitioned subgraphs
        # Architecture: GCN -> Pool -> MLP
        self.param_gen = ParamGenerator()


    def _get_capacity_constrained_hard_s(self, S_soft: torch.Tensor, max_nodes: int) -> torch.Tensor:
        """
        Greedy Capacity Discretization (GCD).
        Converts the soft assignment probabilities S_soft into a hard partition S_hard that satisfies the max_nodes constraint.
        """
        N, k = S_soft.shape
        device = S_soft.device
        
        # 1. Initialize assignment state
        S_hard = torch.zeros((N, k), device=device)
        
        # 2. Prepare ranking data (highly parallelized)
        # Ranks: [N, k], preference order of k partitions for each node (cluster indices)
        # Scores: [N, k], scores of k partitions for each node
        Scores, Ranks = torch.sort(S_soft, dim=1, descending=True) 
        # Complexity: O(Nk log k) — much better than O(Nk log(Nk))

        # Tracking state (assigned nodes / current cluster capacity)
        node_assigned = torch.zeros(N, dtype=torch.bool, device=device)
        cluster_counts = torch.zeros(k, device=device)
        
        # 3. Iterative assignment (up to k rounds)
        # The outer loop is necessary because the capacity constraint must be resolved sequentially.
        for rank_r in range(k):
            if node_assigned.all():
                break

            # Candidate assignments for this round (node i prefers the rank_r-th cluster Ranks[i, r])
            # [N]
            candidate_clusters = Ranks[:, rank_r]
            
            # Unassigned nodes in this round
            unassigned_candidates = torch.where(~node_assigned)[0]
            
            # 4. Parallel assignment logic (CUDA-accelerated)
            # For each cluster, select the highest-priority candidates in this round
            
            for cluster_j in range(k):
                # Unassigned nodes that want to enter cluster_j in this round
                candidates_for_j = unassigned_candidates[
                    candidate_clusters[unassigned_candidates] == cluster_j
                ]
                
                num_available_slots = max_nodes - cluster_counts[cluster_j]
                
                if num_available_slots > 0 and len(candidates_for_j) > 0:
                    # Decide which nodes are accepted (based on scores)
                    # Higher score[i, r] gets higher priority
                    
                    # Scores of these candidates at the current rank
                    # Scores[candidates_for_j, rank_r]
                    candidate_scores = Scores[candidates_for_j, rank_r]
                    
                    # Select up to num_available_slots nodes with highest scores
                    num_to_assign = min(len(candidates_for_j), int(num_available_slots.item()))
                    
                    # Indices of accepted nodes (CUDA topk)
                    _, assigned_local_indices = torch.topk(candidate_scores, k=num_to_assign)
                    
                    # Global indices of accepted nodes
                    assigned_global_indices = candidates_for_j[assigned_local_indices]
                    
                    # 5. Update state (highly parallelized scatter/index_fill)
                    # Update S_hard, node_assigned, cluster_counts
                    
                    # Write S_hard (parallel)
                    S_hard[assigned_global_indices, cluster_j] = 1.0
                    
                    # Mark assigned nodes (parallel)
                    node_assigned[assigned_global_indices] = True
                    
                    # Update capacity (parallel)
                    cluster_counts[cluster_j] += num_to_assign

        return S_hard


    def forward(self, data, k_input, max_nodes_per_partition):
        """
        Full two-stage generator forward pass with Variable-K batching support.
        Args:
            data (Batch data): input graph batch
            k_input: number of partitions per graph
                - If int: assumes all graphs share the same k (or Batch=1)
                - If Tensor: [Batch_Size], per-graph k
        Returns:
            partition_indices (Tensor): [Total_N], local subgraph id for each node
            P (Tensor): initial QAOA circuit parameters [Total_K, 2*depth]
            data (Data): graph data with new attributes (used for loss computation)
        """
        device = data.x.device
        batch_vector = data.batch
        batch_size = batch_vector.max().item() + 1
        
        # -------------------------------------------------
        # 0. Normalize k_input and compute offsets
        # -------------------------------------------------
        if isinstance(k_input, int):
            k_tensor = torch.full((batch_size,), k_input, device=device, dtype=torch.long)
        else:
            k_tensor = k_input if isinstance(k_input, torch.Tensor) else torch.tensor(k_input, device=device)
            
        # Prefix-sum offsets for global subgraph IDs
        # Example: k=[2, 3] -> offsets=[0, 2] -> global_ids: graph0=[0,1], graph1=[2,3,4]
        # cumsum gives [2, 5], subtract k to get [0, 2]
        k_offsets = torch.cumsum(k_tensor, dim=0) - k_tensor 

        # -------------------------------------------------
        # 1. GNN parallel encoding (Partition Phase 1)
        # -------------------------------------------------
        # Process the whole batch at once to utilize GPU parallelism
        H_all = self.partition_gen.topology_encoder(data.x, data.edge_index, data.edge_attr)
        

        # -------------------------------------------------
        # 2. Per-graph loop (Partition Phase 2: GCD & Head)
        # -------------------------------------------------
        local_partition_indices_list = []
        global_subgraph_indices_list = []
        edge_weight_mask_list = []
        
        # This loop is necessary: GCD cannot be parallelized across graphs, and ClusterHead needs per-graph centroids
        for b in range(batch_size):
            # --- 2.1 Slice per-graph data ---
            mask_nodes = (batch_vector == b)
            # Start index for correcting edge_index to local node indices
            start_node_idx = torch.where(mask_nodes)[0][0]
             
            H_b = H_all[mask_nodes]
            k_b = k_tensor[b].item()
            offset_b = k_offsets[b].item()
            
            # --- 2.2 Head (generate soft S) ---
            # S_b: [N_b, k_b]
            S_b, queries_b = self.partition_gen.cluster_head(H_b, k_b)
            
            # --- 2.3 GCD (generate hard S) ---
            with torch.no_grad():
                S_hard_b = self._get_capacity_constrained_hard_s(S_b.detach(), max_nodes_per_partition)
            # --- 2.4 STE (straight-through gradients) ---
            S_ste_b = S_hard_b + (S_b - S_b.detach())
            
            # --- 2.5 Record indices ---
            # local_part_id: [N_b], range 0 ~ k_b-1
            local_part_id = torch.argmax(S_hard_b, dim=1)
            local_partition_indices_list.append(local_part_id)
            
            # global_subgraph_id: [N_b], globally unique
            global_subgraph_id = local_part_id + offset_b
            global_subgraph_indices_list.append(global_subgraph_id)
            
            # --- 2.6 Compute edge weights (edge mask) ---
            
            mask_edges = (batch_vector[data.edge_index[0]] == b)
            global_edge_index_b = data.edge_index[:, mask_edges]
            
            # Map global node indices back to local (0 ~ N_b-1) to match S_ste_b
            local_edge_index_b = global_edge_index_b - start_node_idx
            
            r_local, c_local = local_edge_index_b
            
            # Einsum: [E_b, k] * [E_b, k] -> [E_b]
            # Probability that both endpoints belong to the same partition
            edge_weight_b = torch.einsum('ik,ik->i', S_ste_b[r_local], S_ste_b[c_local])
            edge_weight_mask_list.append(edge_weight_b)

        # -------------------------------------------------
        # 3. Reassemble
        # -------------------------------------------------
        # 3.1 Concatenate partition indices
        partition_indices = torch.cat(local_partition_indices_list, dim=0) # [Total_N]
        
        # 3.2 Concatenate subgraph indices (for ParamGen)
        data.subgraph_batch_index = torch.cat(global_subgraph_indices_list, dim=0) # [Total_N]
        
        # 3.3 Concatenate edge weights
        # Note: concatenation order must strictly match data.edge_index
        # We iterate b=0...B and PyG stacks edges in the same order, so direct cat is safe
        P_hard_mask_all = torch.cat(edge_weight_mask_list, dim=0) # [Total_E]
        
        # 3.4 Update Data object
        data.edge_index_c = data.edge_index
        original_attr = data.edge_attr.squeeze(-1) if data.edge_attr.dim() > 1 else data.edge_attr
        data.edge_weight_c = P_hard_mask_all * original_attr

        # 1. Backup edge weights with gradients
        edge_weight_with_grad = data.edge_weight_c
        
        # 2. Replace edge weights in data with a detached version
        # This ensures P_all computed by param_gen does not include the computation graph of S
        data.edge_weight_c = edge_weight_with_grad.detach()


        # -------------------------------------------------
        # 4. Parameter generation (ParamGen Phase)
        # -------------------------------------------------
        # ParamGenerator is fully parallel; it only uses subgraph_batch_index
        # P_all: [Total_K, 2*depth]
        P_all = self.param_gen(data)
        
        # 4. Restore edge weights with gradients
        # This is important because the returned data may be used to compute the partition loss
        data.edge_weight_c = edge_weight_with_grad

        # Attach parameters to nodes (for loss computation)
        data.node_params = P_all[data.subgraph_batch_index]

        # -------------------------------------------------
        # 5. Return (keep the tuple)
        # -------------------------------------------------
        return partition_indices, P_all, data