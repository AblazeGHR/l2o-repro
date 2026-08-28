import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import os
import json
import pickle
import torch
import numpy as np
import networkx as nx
from pathlib import Path
import psutil
import argparse
from tqdm import tqdm
import gc
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data
from collections import defaultdict
from utils import parse_graph_from_pyg_json, calculate_node_features, partition_to_edge_index_c, partition_to_edge_index_and_weight_c, format_bytes, calculate_sum_neg_weights
import hashlib


BASE_DIR = Path(__file__).resolve().parent.parent


TEST_SET_FILES = {
    # B
    "bqp50-1.txt", "bqp50-2.txt", 
    "bqp100-1.txt", "bqp100-2.txt", 
    "bqp250-1.txt", "bqp250-2.txt", 
    "bqp500-1.txt", "bqp500-2.txt",
    # BE
    "be100.1.txt", "be100.2.txt", 
    "be120.3.1.txt", "be120.3.2.txt", 
    "be120.8.1.txt", "be120.8.2.txt", 
    "be150.3.1.txt", "be150.3.2.txt",
    "be150.8.1.txt", "be150.8.2.txt", 
    "be200.3.1.txt", "be200.3.2.txt", 
    "be200.8.1.txt", "be200.8.2.txt", 
    "be250.1.txt", "be250.2.txt",
    # W
    "g05_60.1.txt", "g05_60.2.txt", 
    "g05_80.1.txt", "g05_80.2.txt", 
    "g05_100.1.txt", "g05_100.2.txt",
    "pm1d_80.1.txt", "pm1d_80.2.txt", 
    "pm1d_100.1.txt", "pm1d_100.2.txt", 
    "pm1s_80.1.txt", "pm1s_80.2.txt",
    "pm1s_100.1.txt", "pm1s_100.2.txt", 
    "pw01_100.1.txt", "pw01_100.2.txt", 
    "pw05_100.1.txt", "pw05_100.2.txt",
    "pw09_100.1.txt", "pw09_100.2.txt", 
    "w01_100.1.txt", "w01_100.2.txt", 
    "w05_100.1.txt", "w05_100.2.txt",
    "w09_100.1.txt", "w09_100.2.txt"
}


def generate_critic_dataset(model_type: str):

    print(f"--- generate_critic_dataset (mode: {model_type}) ---")

    if model_type == "train":

        TRAINING_SET_DIR = BASE_DIR / config.TRAINING_SET_DIR

        CRITIC_DATASET_PATH = BASE_DIR / config.CRITIC_DATASET_PATH

    elif model_type == "test":
        TRAINING_SET_DIR = BASE_DIR / config.TESTING_SET_DIR
        CRITIC_DATASET_PATH = BASE_DIR / config.CRITIC_TEST_DATASET_PATH

    CRITIC_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(CRITIC_DATASET_PATH):
        print(f"Dataset found at {CRITIC_DATASET_PATH}, skipping generation.")
        return

    process = psutil.Process(os.getpid())

    total_available_mem = psutil.virtual_memory().total
    print(f"WSL environment total memory: {format_bytes(total_available_mem)}")

    dataset = []
    feature_cache = {}
    # graph_cache = {}
    edge_index_cache = {}
    edge_attr_cache = {}
    normalized_edge_attr_cache = {}
    skipped_count = 0
    data_points_count = 0
    

    seen_graph_partition_pairs = set()
    duplicate_data_points_skipped = 0 

    print(f"Recursively searching for all JSON files in {TRAINING_SET_DIR}...")
    all_json_files = list(TRAINING_SET_DIR.rglob('*.json'))

    if not all_json_files:
        print("Error: No JSON files found. Please check the TRAINING_SET_DIR path in your config.")
        return
        
    grouped_by_instance = defaultdict(list)
    for file_path in all_json_files:
        instance_dir = file_path.parent.parent 
        grouped_by_instance[instance_dir].append(file_path)

    json_files_to_process = []
    print("Filtering for the latest run for each instance...")
    for instance_dir, files in grouped_by_instance.items():
        if not files:
            continue

        latest_file = max(files, key=lambda f: f.parent.name)
        json_files_to_process.append(latest_file)

    json_files = json_files_to_process
    print(f"Found {len(json_files)} JSON files. Generating dataset for Critic...")
    
    pbar = tqdm(json_files, desc="Processing files")
    for json_path in pbar:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
        except Exception as e:
            print(f"Warning: Could not process {json_path}. Skipping. Error: {e}")
            continue

        graph_file_path = content.get("graph_file", "")
        if model_type == "train" and any(test_name in graph_file_path for test_name in TEST_SET_FILES):
            skipped_count += 1
            continue
        elif model_type == "test" and not any(test_name in graph_file_path for test_name in TEST_SET_FILES):
            skipped_count += 1
            continue

        opt_i = content.get("upper_bound")
        sum_neg_i = content.get("sum_neg")
        if opt_i is None or sum_neg_i is None:
            print(f"Skipping {json_path}: Missing OPT or SumNeg.")
            continue


        for run in content.get('runs', []):
            ratio = run.get('approx_ratio')

            cut_value = run.get('cut_value')

            if ratio is None or not np.isfinite(ratio) or ratio < 0 or ratio > 1: 
                print(f"Warning: Invalid ratio {ratio} in {json_path}. Skipping this data point.")
                continue
            
            step_data_list = run.get('data', [])


            for step_data in step_data_list:
                graph_json = step_data.get('graph')
                if not graph_json: continue
                
                data_id = step_data.get('data_id', 1)

                if data_id > 1:

                    break  

                use_cache = (data_id == 1) and (graph_file_path in feature_cache)
                
                if use_cache:
                    features = feature_cache[graph_file_path]
                    edge_index = edge_index_cache[graph_file_path]
                    normalized_edge_attr = normalized_edge_attr_cache[graph_file_path]
                else:
                    try:
                        _edge_index = torch.tensor(graph_json['edge_index'], dtype=torch.long)
                        _edge_attr = torch.tensor(graph_json['edge_weight'], dtype=torch.float32).unsqueeze(1)
                        
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

                        
   
                        graph = parse_graph_from_pyg_json(graph_json)
                        features = calculate_node_features(graph)


                        if data_id == 1:  
                            feature_cache[graph_file_path] = features
                            edge_index_cache[graph_file_path] = edge_index
                            normalized_edge_attr_cache[graph_file_path] = normalized_edge_attr


                    except Exception as e:
                        print(f"Error parsing graph in step: {e}")
                        continue

                partitions = step_data.get('partition')
                if not partitions: continue
                
                num_nodes = features.shape[0]
                node_to_part_id = torch.full((num_nodes,), -1)
                
                valid_partition = True
                for i, part in enumerate(partitions):
                    if part: 
                        part_tensor = torch.tensor(part)
                        if part_tensor.max() >= num_nodes:
                            print(f"Warning: Partition index out of bounds in {json_path}. Skipping this run.")
                            valid_partition = False
                            break
                        node_to_part_id[part_tensor] = i
                
                if not valid_partition:
                    continue 

                u_nodes = edge_index[0] 
                v_nodes = edge_index[1] 
                
                part_id_u = node_to_part_id[u_nodes] 
                part_id_v = node_to_part_id[v_nodes] 
                

                keep_mask_c = (part_id_u == part_id_v) & (part_id_u != -1) # [2*NumEdges]
                

                coeff_mask_c = torch.zeros(edge_index.shape[1], dtype=torch.float32, device=features.device)
                coeff_mask_c[keep_mask_c] = 1.0

                edge_index_c = edge_index
                

                normalized_edge_weight_c = normalized_edge_attr.squeeze(-1) * coeff_mask_c
   
                raw_params = step_data.get('init_gammas_betas') 
                
                if raw_params is None:
                    continue 
                
                P_tensor = torch.tensor(raw_params, dtype=torch.float32)
                k_subgraphs = P_tensor.shape[0]
                depth = P_tensor.shape[2]
                
                P_flat = P_tensor.view(k_subgraphs, -1)

                valid_mask = node_to_part_id != -1
                
                node_params = torch.zeros((num_nodes, P_flat.shape[1]), dtype=torch.float32)
                
                node_params[valid_mask] = P_flat[node_to_part_id[valid_mask]]

                r_val = step_data.get('const')

                alpha_Q = (cut_value - r_val) / (opt_i - r_val)


                data = Data(
                    x=features,
                    
                    # Graph Encoder Inputs
                    edge_index=edge_index,
                    edge_attr=normalized_edge_attr,
                    
                    # Partition Encoder Inputs
                    edge_index_c=edge_index_c,
                    edge_weight_c=normalized_edge_weight_c,
                    
                    # QAOA Params Encoder Inputs
                    node_params=node_params,
                    
                    # Scalars for Loss Calculation (Wrapped in Tensor for batching)
                    r=torch.tensor([r_val], dtype=torch.float32),
                    sum_neg=torch.tensor([sum_neg_i], dtype=torch.float32),
                    OPT=torch.tensor([opt_i], dtype=torch.float32),
                    
                    # Label
                    y=torch.tensor([ratio], dtype=torch.float32),

                    # alpha_Q
                    alpha_Q=torch.tensor([alpha_Q], dtype=torch.float32)

                )


                edge_bytes = edge_index.cpu().numpy().tobytes()

                r_bytes = str(r_val).encode('utf-8')

                partition_bytes = normalized_edge_weight_c.cpu().numpy().tobytes()


                hash_input = edge_bytes + r_bytes + partition_bytes

                content_hash = hashlib.md5(hash_input).hexdigest()

                if content_hash in seen_graph_partition_pairs:
                    duplicate_data_points_skipped += 1
                    continue

                seen_graph_partition_pairs.add(content_hash)

                dataset.append(data)
                data_points_count += 1



        used_mem = process.memory_info().rss

        available_mem = psutil.virtual_memory().available
        
        mem_str = (f"Mem (Used/Avail): {format_bytes(used_mem)} / "
                   f"{format_bytes(available_mem)}")
        
        pbar.set_postfix_str(mem_str)

    
    pbar.close() 

    if model_type == "train":
        print(f"Total {len(TEST_SET_FILES)} test instances skipped, JSON files skipped: {skipped_count}")
    elif model_type == "test":
        print(f"Total {len(TEST_SET_FILES)} test instances processed, JSON files skipped: {skipped_count}")


    print(f"Total duplicate (graph, partition) pairs skipped: {duplicate_data_points_skipped}")

    print(f"Total data points generated: {data_points_count}")


    print("Clearing graph, feature, and sum_neg caches to free up memory before saving...")
    
    #graph_cache.clear()
    feature_cache.clear()
    edge_index_cache.clear()
    edge_attr_cache.clear()
    normalized_edge_attr_cache.clear()
    seen_graph_partition_pairs.clear()
    gc.collect()
    
    print("Caches cleared. Now saving the dataset...")
    
    with open(CRITIC_DATASET_PATH, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset with {len(dataset)} samples saved to {CRITIC_DATASET_PATH}")


def generate_actor_graph_dataset(model_type: str):

    print(f"--- generate_actor_graph_dataset (mode: {model_type}) ---")

    if model_type == "train":
        INPUT_DIR = BASE_DIR / config.TRAINING_SET_DIR
        OUTPUT_PATH = BASE_DIR / config.ACTOR_GRAPH_DATASET_PATH
        output_dir = os.path.join(ROOT, "data", "datasets", "training-set", "instances")
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, "train_instances_name.txt")
        print(f"\nWriting train instances name to {output_file_path} ...")
    elif model_type == "test":
        INPUT_DIR = BASE_DIR / config.TESTING_SET_DIR
        OUTPUT_PATH = BASE_DIR / config.ACTOR_TEST_GRAPH_DATASET_PATH
        output_dir = os.path.join(ROOT, "data", "datasets", "testing-set", "instances")
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, "test_instances_name.txt")
        print(f"\nWriting test instances name to {output_file_path} ...")
    else:
        raise ValueError("Invalid model type. Choose 'train' or 'test'")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(OUTPUT_PATH):
        print(f"Actor graph dataset found at {OUTPUT_PATH}, skipping generation.")
        return


    process = psutil.Process(os.getpid())
    total_available_mem = psutil.virtual_memory().total
    print(f"WSL environment total memory: {format_bytes(total_available_mem)}")
    # =========================================================================

    dataset = []
    feature_cache = {}
    # graph_cache = {}
    edge_index_cache = {}
    edge_attr_cache = {}
    normalized_edge_attr_cache = {}
    skipped_count = 0
    data_points_count = 0
    
    seen_graph_partition_pairs = set()
    duplicate_data_points_skipped = 0 
    # =========================================================================

    print(f"Recursively searching for all JSON files in {INPUT_DIR}...")
    all_json_files = list(INPUT_DIR.rglob('*.json'))

    if not all_json_files:
        print("Error: No JSON files found. Please check the INPUT_DIR path in your config.")
        return
        

    grouped_by_instance = defaultdict(list)
    for file_path in all_json_files:
        instance_dir = file_path.parent.parent 
        grouped_by_instance[instance_dir].append(file_path)

    json_files_to_process = []
    print("Filtering for the latest run for each instance...")
    for instance_dir, files in grouped_by_instance.items():
        if not files:
            continue

        latest_file = max(files, key=lambda f: f.parent.name)
        json_files_to_process.append(latest_file)

    json_files = json_files_to_process
    print(f"Found {len(json_files)} JSON files. Generating dataset for Actor ...")
    
    pbar = tqdm(json_files, desc="Processing files")
    for json_path in pbar:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
        except Exception as e:
            print(f"Warning: Could not process {json_path}. Skipping. Error: {e}")
            continue

        graph_file_path = content.get("graph_file", "")

        if model_type == "train" and any(test_name in graph_file_path for test_name in TEST_SET_FILES):
            skipped_count += 1
            continue
        elif model_type == "test" and not any(test_name in graph_file_path for test_name in TEST_SET_FILES):
            skipped_count += 1
            continue


        opt_i = content.get("upper_bound")
        sum_neg_i = content.get("sum_neg")
        if opt_i is None or sum_neg_i is None:
            print(f"Skipping {json_path}: Missing OPT or SumNeg.")
            continue


        for run in content.get('runs', []):
            

            step_data_list = run.get('data', [])


            for step_data in step_data_list:
                graph_json = step_data.get('graph')
                if not graph_json: continue
                

                data_id = step_data.get('data_id', 1)
                

                if data_id > 1:

                    break  

                use_cache = (data_id == 1) and (graph_file_path in feature_cache)
                
                if use_cache:
                    features = feature_cache[graph_file_path]
                    edge_index = edge_index_cache[graph_file_path]
                    normalized_edge_attr = normalized_edge_attr_cache[graph_file_path]
                else:
                    try:
                        _edge_index = torch.tensor(graph_json['edge_index'], dtype=torch.long)
                        _edge_attr = torch.tensor(graph_json['edge_weight'], dtype=torch.float32).unsqueeze(1)
                        
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

                        graph = parse_graph_from_pyg_json(graph_json)
                        features = calculate_node_features(graph)

                        if data_id == 1: 
                            feature_cache[graph_file_path] = features
                            edge_index_cache[graph_file_path] = edge_index
                            normalized_edge_attr_cache[graph_file_path] = normalized_edge_attr


                    except Exception as e:
                        print(f"Error parsing graph in step: {e}")
                        continue

                r_val = step_data.get('const', 0.0)

                data = Data(
                    x=features,
                    
                    # Graph Encoder Inputs
                    edge_index=edge_index,
                    edge_attr=normalized_edge_attr,
                    
                    # Scalars for Loss Calculation (Wrapped in Tensor for batching)
                    r=torch.tensor([r_val], dtype=torch.float32),
                    sum_neg=torch.tensor([sum_neg_i], dtype=torch.float32),
                    OPT=torch.tensor([opt_i], dtype=torch.float32),
                )


                edge_bytes = edge_index.cpu().numpy().tobytes()

                r_bytes = str(r_val).encode('utf-8')

                hash_input = edge_bytes + r_bytes

                content_hash = hashlib.md5(hash_input).hexdigest()

                if content_hash in seen_graph_partition_pairs:
                    duplicate_data_points_skipped += 1
                    continue

                seen_graph_partition_pairs.add(content_hash)

                dataset.append(data)
                data_points_count += 1

        used_mem = process.memory_info().rss

        available_mem = psutil.virtual_memory().available
        
        mem_str = (f"Mem (Used/Avail): {format_bytes(used_mem)} / "
                   f"{format_bytes(available_mem)}")
        
        pbar.set_postfix_str(mem_str)

    
    pbar.close() 

    if model_type == "train":
        print(f"Total {len(TEST_SET_FILES)} test instances skipped, JSON files skipped: {skipped_count}")
    elif model_type == "test":
        print(f"Total {len(TEST_SET_FILES)} test instances processed, JSON files skipped: {skipped_count}")

    print(f"Total duplicate (graph, partition) pairs skipped: {duplicate_data_points_skipped}")
    print(f"Total data points generated: {data_points_count}")
    print("Clearing graph, feature, and sum_neg caches to free up memory before saving...")
    
    #graph_cache.clear()
    feature_cache.clear()
    edge_index_cache.clear()
    edge_attr_cache.clear()
    normalized_edge_attr_cache.clear()
    seen_graph_partition_pairs.clear()
    gc.collect()
    
    print("Caches cleared. Now saving the dataset...")
    # ==========================================================
    
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Actor graph dataset saved to {OUTPUT_PATH}")


class CriticDataset(Dataset):
    def __init__(self, dataset_path):
        if not os.path.exists(dataset_path):
            print(f"Critic dataset file not found: {dataset_path}")
            print("Please run 'python data.py --type critic --model [train/test]' to generate it.")
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        with open(dataset_path, 'rb') as f:
            self.data = pickle.load(f)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ActorGraphDataset(Dataset):
    def __init__(self, dataset_path):
        if not os.path.exists(dataset_path):
            print(f"Actor graph dataset file not found: {dataset_path}")
            print("Please run 'python data.py --type actor --model [train/test]' to generate it.")
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        with open(dataset_path, 'rb') as f:
            self.data = pickle.load(f)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ===============================================================

# ===============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Datasets for Actor or Critic")
    parser.add_argument("--type", type=str, default="critic", 
                        help="Dataset type to generate: 'critic' (graphs+partitions) or 'actor' (graphs)")
    parser.add_argument("--model", type=str, default="train", 
                        help="Model type: 'train' (skip test files) or 'test' (only test files)")
    args = parser.parse_args()

    if args.type == 'critic':
        generate_critic_dataset(model_type=args.model)
    
    elif args.type == 'actor':
        generate_actor_graph_dataset(model_type=args.model)
        
    else:
        print(f"Error: Unknown dataset type '{args.type}'. Choose 'critic' or 'actor'.")