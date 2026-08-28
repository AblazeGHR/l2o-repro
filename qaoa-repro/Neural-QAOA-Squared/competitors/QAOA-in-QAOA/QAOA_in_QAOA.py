import numpy as np
import pennylane as qml
import math
from tqdm import tqdm
import json
import os
import importlib.util
import argparse
from utilities import *
from QAOA import *
from pathlib import Path
from datetime import datetime
import time
from scipy.interpolate import interp1d
from scipy.fft import dct, idct, dst, idst
import csv
import networkx as nx
# --- Helper Functions ---

def get_timestamp():
    """Return a formatted timestamp string for the current time."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def format_time(seconds):
    """Format a duration in seconds into a human-readable string."""
    if seconds is None:
        return None
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds * 1000) % 1000)
    
    if h > 0: return f"{h}h {m}m {s}s"
    if m > 0: return f"{m}m {s}s"
    if s > 0: return f"{s}s {ms}ms"
    return f"{ms}ms"

def get_output_path(data_path_str, experiment_char, depth, sub_size, policy):
    """Build a timestamped output path from the input path and parameters."""
    experiment_folder = 'preliminary' if experiment_char == 'p' else 'main'
    timestamp_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    p = Path(data_path_str)
    dataset_folder = p.parts[-3]
    algorithm_folder = "QAOA-in-QAOA" + f"_d{depth}_s{sub_size}_{policy}"
    instance_name = p.stem
    
    output_dir = Path("result") / experiment_folder / dataset_folder / algorithm_folder / instance_name / timestamp_folder
    # Include QAOA-specific parameters in the filename
    output_filename = f"{instance_name}_qaoa_d{depth}_s{sub_size}_{policy}_result.json"
    
    return output_dir / output_filename

def load_graph_data(data_path):
    """Load graph data based on the file extension."""
    try:
        file_extension = os.path.splitext(data_path)[1]
        
        if file_extension == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            n_v = graph_data['n_v']
            edges = graph_data['edges']
            n_e = len(edges)
            graph_data['n_e'] = n_e
        
        elif file_extension == '.txt':
            with open(data_path, 'r') as f:
                lines = f.readlines()
                header = lines[0].strip().split()
                n_v = int(header[0])
                n_e = int(header[1])
                edges = []
                for line in lines[1:]:
                    if line.strip():
                        parts = line.strip().split()
                        u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
                        edges.append([u - 1, v - 1, w])
            graph_data = {'n_v': n_v, 'n_e': n_e}
        else:
            print(f"Error: Unsupported file format '{file_extension}'.")
            return None, None
        return graph_data, edges
    except Exception as e:
        print(f"Error: Failed to read or parse file {data_path}: {e}")
        return None, None

def flip_bitstring(s):
    """Flip a bitstring (e.g., '0110' -> '1001')."""
    return "".join(['1' if c == '0' else '0' for c in s])

def calculate_sum_neg_weights(edges):
    """
    Compute the sum of weights over all negatively weighted edges (SumNeg).
    
    Args:
        edges: Edge list, each element is [u, v, w].
    
    Returns:
        float: Sum of all negative edge weights.
    """
    sum_neg = 0.0
    for edge in edges:
        if len(edge) >= 3:
            weight = float(edge[2])
            if weight < 0:
                sum_neg += weight
    return sum_neg

def reconstruct_qaoa_solution(graph_n_v, qaoa_sols):
    """
    [Corrected function]
    Reconstruct the final partition assignment from the hierarchical solutions of QAOA².
    This function uses the correct top-down, level-by-level propagation logic.

    Args:
        graph_n_v (int): Total number of vertices in the original graph.
        qaoa_sols (dict): The 'sol' dictionary loaded from QAOA² results; keys must be integers.

    Returns:
        str: The reconstructed full 0/1 bitstring.
    """
    num_levels = len(qaoa_sols)
    if num_levels == 0:
        return ""

    final_solution_map = {}
    
    # Start from the highest level (the last level)
    final_level = num_levels - 1
    
    # final_partitions stores the final assignment of each “super-node” at the current level
    # Initialize with the partition assignment at the highest level
    final_partitions = {v: s for v, s in zip(qaoa_sols[final_level]['v'], qaoa_sols[final_level]['sol'])}

    # Trace down level by level starting from the second-to-last level
    for level in range(final_level - 1, -1, -1):
        new_partitions = {}
        # Iterate over each subgraph at this level (these are the “super-nodes” of the level above)
        for i, subgraph_nodes in enumerate(qaoa_sols[level]['v']):
            # Get this subgraph's assignment at the upper level ('0' or '1')
            parent_partition = final_partitions.get(i)
            
            # Get the internal partition string of this subgraph
            internal_partition_str = qaoa_sols[level]['sol'][i]

            # If the upper-level assignment is '1', flip the internal assignment
            if parent_partition == '1':
                internal_partition_str = flip_bitstring(internal_partition_str)
            
            # Map the final assignment back to each node in this subgraph
            for node_index, node_partition in zip(subgraph_nodes, internal_partition_str):
                new_partitions[node_index] = node_partition
        
        # Update final_partitions to prepare for the next (lower) level
        final_partitions = new_partitions

    # Convert the mapping into the final bitstring
    final_solution_array = ['0'] * graph_n_v
    for node_idx, partition in final_partitions.items():
        if node_idx < graph_n_v:
            final_solution_array[node_idx] = partition
            
    return "".join(final_solution_array)

def get_interp_params(prev_opt_params, new_depth):
    """
    prev_opt_params: Shape (2, old_depth)
    new_depth: int, usually old_depth + 1
    Returns: new_params of shape (2, new_depth)
    """
    prev_opt_params = np.array(prev_opt_params)
    
    if prev_opt_params.ndim == 1:
        old_depth = prev_opt_params.shape[0] // 2
        prev_opt_params = prev_opt_params.reshape(2, old_depth)
    else:
        _, old_depth = prev_opt_params.shape
    
    if old_depth == 1:
        new_params = np.tile(prev_opt_params, (1, new_depth))
        return new_params

    old_indices = np.linspace(0, 1, old_depth)
    new_indices = np.linspace(0, 1, new_depth)
    
    new_params = np.zeros((2, new_depth))
    for i in range(2): # 0 for gamma, 1 for beta
        f = interp1d(old_indices, prev_opt_params[i], kind='linear', fill_value="extrapolate")
        new_params[i] = f(new_indices)
        
    return new_params


# --- Core Algorithm ---

def qaoa_square(data_path:str, depth:int=1, sub_size:int=10, partition_policy:str='random', init_gammas=None, init_betas=None):
    '''
    The QAOA Squared divide-and-conquer framework remains unchanged.
    '''
    graph_data, edges = load_graph_data(data_path)
    if not graph_data:
        return None, None
    n_v = graph_data['n_v']
    
    G = Graph(v=list(range(n_v)), edges=edges)

    print(f"[{get_timestamp()}] Initial graph: {n_v} vertices, {len(edges)} edges")
    print(f"[{get_timestamp()}] Parameters: depth={depth}, sub_size={sub_size}, policy='{partition_policy}'")
    
    const = 0
    sols = {}
    level = 0 
    
    while G.n_v > sub_size:
        print(f"[{get_timestamp()}] [Level {level}] Current graph: {G.n_v} vertices, {len(G.e)} edges")
        sols[level] = {}
        H, init_gammas_betas = G.graph_partition(n=sub_size, policy=partition_policy, depth=depth)
        
        print(f"   Partitioned into {len(H)} subgraphs:")
        for i, h in enumerate(H):
            print(f"      Subgraph {i+1}: {h.n_v} vertices, {len(h.e)} edges")
        obj = []
        sol = []
        
        with tqdm(total=len(H), desc=f'Level {level} subgraphs', leave=False, ascii=False) as pbar:
            for idx, H_sub in enumerate(H):
                const_temp = 0.5 * sum([x[2] for x in H_sub.e])
                # Use the pre-generated initialization parameters for this subgraph
                curr_init_gammas = init_gammas_betas[idx][0, :]
                curr_init_betas = init_gammas_betas[idx][1, :]

                # ret = qaoa(H_sub, const=const_temp, n_layers=depth, sample_method='max')
                ret = qaoa(H_sub, const=const_temp, n_layers=depth, sample_method='max', 
                            init_gammas=curr_init_gammas, init_betas=curr_init_betas)
                obj_value = float(ret[0]) if not isinstance(ret[0], (int, float)) else ret[0]
                obj.append(obj_value)
                sol.append(ret[1])
                pbar.update(1)

        sols[level]['sol'] = sol
        sols[level]['v'] = [h.v for h in H]
        n_sub = len(H)


        # TODO: record G, H, (init_gammas, init_betas) for each H_sub, const 
        
        adjoint = np.zeros((n_sub, n_sub))
        for i in range(n_sub):
            for j in range(i+1, n_sub):
                w_pos, w_neg = 0, 0
                for x in range(H[i].n_v):
                    for y in range(H[j].n_v):
                        m, n = H[i].v[x], H[j].v[y]
                        edge_weight = G.adj[m][n]
                        w_pos += (sol[i][x]!=sol[j][y]) * edge_weight
                        w_neg += (sol[i][x]==sol[j][y]) * edge_weight
                adjoint[i][j]= w_neg-w_pos
                adjoint[j][i]= w_neg-w_pos
                const += w_pos
            const += obj[i]
        G = Graph(v=list(range(n_sub)), adjoint=adjoint)
        level += 1

    print(f"[{get_timestamp()}] [Final level {level}] Solving remaining graph ({G.n_v} vertices)...")
    # Even if we stop further subdivision, still call graph_partition to obtain
    # consistent outputs H and init_gammas_betas.
    H, init_gammas_betas = G.graph_partition(
        n=sub_size,
        policy=partition_policy,
        depth=depth
    )
    const_temp = 0.5 * sum([x[2] for x in G.e])
    # ret = qaoa(G, const = const+const_temp, n_layers=depth)
    ret = qaoa(G, const = const+const_temp, n_layers=depth, 
               init_gammas=init_gammas_betas[0][0, :], init_betas=init_gammas_betas[0][1, :])
    
    sols[level] = {}
    sols[level]['sol'] = ret[1]
    sols[level]['v'] = G.v

    # TODO: record G, H, (init_gammas, init_betas) for each H_sub, const 

    return ret[0].item(), sols

# --- Main Execution Block ---

def _load_bruteforce_solver():
    """Load brute_force() from brute-force.py (hyphenated filename)."""
    module_path = Path(__file__).with_name('brute-force.py')
    spec = importlib.util.spec_from_file_location("bruteforce_module", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load brute-force.py module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, 'brute_force')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solve a graph instance multiple times using the QAOA² algorithm.")
    parser.add_argument('--data_path', type=str, required=True, help='Input graph data file path (.json or .txt).')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs for the same instance (default: 1).')
    parser.add_argument('--experiment', type=str, choices=['p', 'm'], required=True, help="Experiment type: 'p' (preliminary) or 'm' (main).")
    parser.add_argument('--optimal_value', type=float, default=None, help="Known optimal cut value for this instance (optional), used to compute approximation ratio.")
    parser.add_argument('--depth', type=int, default=1, help='QAOA circuit depth p (default: 1).')
    parser.add_argument('--sub_size', type=int, default=10, help='Maximum allowed subgraph size (i.e., number of qubits) (default: 10).')
    parser.add_argument('--policy', type=str, default='random', help="Graph partition policy (default: random).")
    parser.add_argument('--base', type=str, choices=['bf', 'qaoa'], default='qaoa', help="Base subproblem solver: 'bf' for brute force, 'qaoa' for QAOA.")
    # nargs='+' accepts multiple values, e.g., --init_gammas 0.1 0.2
    parser.add_argument('--init_gammas', type=float, nargs='+', default=None, help='Initial gamma list for QAOA (length must equal depth).')
    parser.add_argument('--init_betas', type=float, nargs='+', default=None, help='Initial beta list for QAOA (length must equal depth).')
    args = parser.parse_args()

    if args.init_gammas is not None:
        if len(args.init_gammas) != args.depth:
            print(f"Error: Number of provided gamma values ({len(args.init_gammas)}) does not match depth ({args.depth}).")
            exit()
    if args.init_betas is not None:
        if len(args.init_betas) != args.depth:
            print(f"Error: Number of provided beta values ({len(args.init_betas)}) does not match depth ({args.depth}).")
            exit()

    graph_data, edges = load_graph_data(args.data_path)
    if not graph_data: exit()

    n_v, n_e = graph_data['n_v'], graph_data['n_e']
    
    # Calculate SumNeg once at the beginning (used for ratio calculation)
    sum_neg = calculate_sum_neg_weights(edges) if edges else 0.0
    
    all_run_results, cut_values_list, times_in_seconds_list = [], [], []
    total_start_time = time.time()

    print("=" * 60)
    print(f"Start processing instance: {args.data_path}")
    # ... (print statements for info)

    # Choose base solver by rebinding module-level qaoa used inside qaoa_square
    if args.base == 'bf':
        qaoa = _load_bruteforce_solver()
    else:
        # use qaoa imported from QAOA.py
        pass
    

    for i in range(args.runs):
        run_start_time = time.time()
        print(f"\n--- Run {i+1}/{args.runs} started ---")
        
        value, sols = qaoa_square(
            data_path=args.data_path, 
            depth=args.depth, 
            sub_size=args.sub_size,
            partition_policy=args.policy,
            init_gammas=args.init_gammas, 
            init_betas=args.init_betas
        )
        
        if value is None:
            print(f"--- Run {i+1} failed, skipping ---")
            continue

        solution_str = reconstruct_qaoa_solution(n_v, sols)
        run_elapsed_time = time.time() - run_start_time
        cut_values_list.append(value)
        times_in_seconds_list.append(run_elapsed_time)
        if args.optimal_value is not None and args.optimal_value > 0:
            try:
                denominator = args.optimal_value - sum_neg
                approx_ratio = float(value - sum_neg) / float(denominator)
            except Exception:
                approx_ratio = None
        else:
            approx_ratio = None
        
        all_run_results.append({
            "run_id": i + 1,
            "cut_value": float(value),
            "approx_ratio": float(approx_ratio) if approx_ratio is not None else None,
            "solution": solution_str,
            "time": format_time(run_elapsed_time)
        })
        if approx_ratio is not None:
            print(f"--- Run {i+1} completed, time: {format_time(run_elapsed_time)}, max-cut value: {value:.2f}, approx. ratio: {approx_ratio:.4f} ---")
        else:
            print(f"--- Run {i+1} completed, time: {format_time(run_elapsed_time)}, max-cut value: {value:.2f} ---")

    # Perform statistical calculations
    best_approximation_ratio, average_approximation_ratio = None, None
    std_approximation_ratio, median_approximation_ratio = None, None
    
    if args.runs > 0 and cut_values_list:
        cut_stats = np.array(cut_values_list)
        time_stats = np.array(times_in_seconds_list)
        best_run_index = np.argmax(cut_stats)
        
        best_cut_value = cut_stats[best_run_index]
        best_solution = all_run_results[best_run_index]['solution']
        min_cut, max_cut, median_cut, average_cut = np.min(cut_stats), np.max(cut_stats), np.median(cut_stats), np.mean(cut_stats)
        std_cut = np.std(cut_stats) if len(cut_stats) > 1 else 0.0
        average_time_sec = np.mean(time_stats)
        std_time_sec = np.std(time_stats) if len(time_stats) > 1 else 0.0

        if args.optimal_value is not None and args.optimal_value > 0:
            # Calculate denominator: OPT(I) - SumNeg(I)
            denominator = args.optimal_value - sum_neg
            
            # Check if denominator is valid (not zero or too small)
            if abs(denominator) < 1e-9:
                print(f"Warning: Denominator is zero or too small for ratio calculation (OPT={args.optimal_value}, SumNeg={sum_neg}). Skipping ratio calculation.")
                best_approximation_ratio, average_approximation_ratio = None, None
                std_approximation_ratio, median_approximation_ratio = None, None
            else:
                # Calculate adjusted approximation ratio using the formula from data.py:
                # ρ' = (A(I) - SumNeg(I)) / (OPT(I) - SumNeg(I))
                ratios = (cut_stats - sum_neg) / denominator
                
                # Filter out invalid ratios (should be between 0 and 1)
                valid_ratios = ratios[(ratios >= 0) & (ratios <= 1) & np.isfinite(ratios)]
                
                if len(valid_ratios) > 0:
                    best_approximation_ratio = np.max(valid_ratios)
                    average_approximation_ratio = np.mean(valid_ratios)
                    std_approximation_ratio = np.std(valid_ratios) if len(valid_ratios) > 1 else 0.0
                    median_approximation_ratio = np.median(valid_ratios)
                else:
                    print("Warning: No valid ratios found after filtering.")
                    best_approximation_ratio, average_approximation_ratio = None, None
                    std_approximation_ratio, median_approximation_ratio = None, None
    else:
        # Handle cases with no successful runs
        best_cut_value, best_solution, min_cut, max_cut, median_cut, average_cut, std_cut = [None] * 7
        average_time_sec, std_time_sec = None, None
        
    final_json = {
        "algorithm": "QAOA-in-QAOA",
        "graph_file": args.data_path,
        "graph_info": {"n_v": n_v, "n_e": n_e},
        "parameters": {"depth": args.depth, "sub_size": args.sub_size, "policy": args.policy, "base": args.base, "init_gammas": args.init_gammas, "init_betas": args.init_betas},
        "best_cut_value": float(best_cut_value) if best_cut_value is not None else None,
        "best_solution": best_solution,
        "min_cut": float(min_cut) if min_cut is not None else None,
        "max_cut": float(max_cut) if max_cut is not None else None,
        "median_cut": float(median_cut) if median_cut is not None else None,
        "average_cut": float(average_cut) if average_cut is not None else None,
        "std_cut": float(std_cut) if std_cut is not None else None,
        "upper_bound": args.optimal_value,
        "average_time": format_time(average_time_sec),
        "std_time": format_time(std_time_sec),
        "best_approximation_ratio": float(best_approximation_ratio) if best_approximation_ratio is not None else None,
        "average_approximation_ratio": float(average_approximation_ratio) if average_approximation_ratio is not None else None,
        "std_approximation_ratio": float(std_approximation_ratio) if std_approximation_ratio is not None else None,
        "median_approximation_ratio": float(median_approximation_ratio) if median_approximation_ratio is not None else None,
        "runs": all_run_results
    }
    
    output_path = get_output_path(args.data_path, args.experiment, args.depth, args.sub_size, args.policy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 60)
    print("All runs finished!")
    print(f"Results saved to: {output_path}")
    print("=" * 60)