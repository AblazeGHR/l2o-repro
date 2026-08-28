import argparse
import subprocess
import json
import os
import time
import multiprocessing
import datetime
from pathlib import Path
from queue import Empty

# Global lock to keep console output ordered
print_lock = multiprocessing.Lock()

def get_timestamp():
    """Return a formatted timestamp string for the current time."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def format_timedelta(seconds):
    """Format seconds as H:M:S."""
    return str(datetime.timedelta(seconds=int(seconds)))

def load_optimal_values(file_path):
    """Load an optimal-value lookup table from a JSON file."""
    if not file_path or not Path(file_path).is_file():
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Warning: Failed to load optimal values file: {e}")
        return {}

def worker_process(worker_id, gpu_id, task_queue, args, optimal_values, logs_dir, 
                   total_tasks, shared_counter, global_start_time):
    """
    Worker process function.

    Added: total_tasks, shared_counter, global_start_time.
    """
    algo_path = Path(args.algorithm)
    
    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Bind a CPU core
    cpu_core_id = worker_id 

    while True:
        try:
            instance_path = task_queue.get_nowait()
        except Empty:
            break

        instance_name = instance_path.name
        instance_stem = instance_path.stem
        log_file = logs_dir / f"{instance_stem}.log"

        # Build command
        command = [
            "taskset", "-c", str(cpu_core_id),
            "python", str(algo_path),
            "--data_path", str(instance_path),
            "--experiment", args.experiment,
            "--runs", str(args.runs)
        ]

        # Add optimal value (if provided)
        optimal_val = optimal_values.get(instance_stem)
        if optimal_val is not None:
            command.extend(["--optimal_value", str(optimal_val)])
        
        # Add QAOA-specific arguments
        if 'competitors/QAOA-in-QAOA/QAOA_in_QAOA.py' in str(algo_path).replace("\\", "/") or algo_path.name == 'QAOA_in_QAOA.py':
            if args.depth is not None: command.extend(['--depth', str(args.depth)])
            if args.sub_size is not None: command.extend(['--sub_size', str(args.sub_size)])
            if args.policy is not None: command.extend(['--policy', str(args.policy)])
            if args.base is not None: command.extend(['--base', str(args.base)])

        # Run command
        start_time = time.time()
        try:
            with open(log_file, "w", encoding="utf-8") as f_log:
                f_log.write(f"Command: {' '.join(command)}\n")
                f_log.write(f"GPU: {gpu_id}, CPU Core: {cpu_core_id}\n")
                f_log.write("-" * 40 + "\n")
                f_log.flush()
                
                subprocess.run(
                    command, 
                    env=env,
                    stdout=f_log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    text=True
                )
            
            elapsed = time.time() - start_time
            
            # --- Progress & ETA core section ---
            with shared_counter.get_lock():
                shared_counter.value += 1
                current_completed = shared_counter.value
            
            # Compute global elapsed time and ETA
            total_elapsed = time.time() - global_start_time
            avg_time_per_task = total_elapsed / current_completed
            remaining_tasks = total_tasks - current_completed
            eta_seconds = avg_time_per_task * remaining_tasks
            eta_str = format_timedelta(eta_seconds)
            progress_pct = (current_completed / total_tasks) * 100
            
            with print_lock:
                # Print format: [progress | ETA] [GPU info] filename (elapsed)
                print(f"[{progress_pct:5.1f}% | ETA: {eta_str}] [GPU-{gpu_id}] ✅ Done: {instance_name} ({elapsed:.1f}s)")

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            
            # Update the counter even on failure; otherwise progress will stall
            with shared_counter.get_lock():
                shared_counter.value += 1
                current_completed = shared_counter.value
            
            # Compute ETA (failures also consume time)
            total_elapsed = time.time() - global_start_time
            avg_time_per_task = total_elapsed / current_completed
            remaining_tasks = total_tasks - current_completed
            eta_str = format_timedelta(avg_time_per_task * remaining_tasks)

            with print_lock:
                print(f"[{current_completed}/{total_tasks} | ETA: {eta_str}] [GPU-{gpu_id}] ❌ Failed: {instance_name} (Code: {e.returncode})")
                
        except Exception as e:
            with print_lock:
                print(f"❌ [GPU-{gpu_id}] System error: {instance_name} - {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Run graph algorithm scripts in parallel (GPU-accelerated).")
    parser.add_argument('--algorithm', type=str, required=True, help="Path to the algorithm script")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--experiment', type=str, required=True, choices=['p', 'm'])
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--optimal_values_file', type=str, default=None)
    
    # QAOA parameters
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--sub_size', type=int, default=None)
    parser.add_argument('--policy', type=str, default=None)
    parser.add_argument('--base', type=str, choices=['bf', 'qaoa'], default=None)

    # GPU list
    parser.add_argument('--gpus', type=str, default="0,1,2,3", help="Comma-separated GPU IDs")
    
    args = parser.parse_args()

    algo_path = Path(args.algorithm)
    dataset_path = Path(args.dataset_path)
    if not algo_path.is_file() or not dataset_path.is_dir():
        print("❌ Error: Invalid algorithm file or dataset path.")
        return

    logs_dir = Path("batch_logs") / f"{time.strftime('%Y%m%d_%H%M%S')}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect tasks
    instance_files = sorted(
        [f for f in dataset_path.glob('*.txt')] + 
        [f for f in dataset_path.glob('*.json')]
    )
    if not instance_files:
        print("⚠️  Warning: No instance files found.")
        return
    
    total_tasks = len(instance_files)
    optimal_values = load_optimal_values(args.optimal_values_file)
    gpu_list = [int(x.strip()) for x in args.gpus.split(',') if x.strip()]
    num_workers = len(gpu_list)
    
    print(f"[{get_timestamp()}] 🚀 Starting parallel processing | Total tasks: {total_tasks} | GPUs: {gpu_list}")
    print(f"Logs directory: {logs_dir}")
    print("-" * 60)

    task_queue = multiprocessing.Queue()
    for f in instance_files:
        task_queue.put(f)

    # Shared counter (int, initial value 0)
    shared_counter = multiprocessing.Value('i', 0)
    
    # Global start time
    global_start_time = time.time()

    processes = []
    for i, gpu_id in enumerate(gpu_list):
        p = multiprocessing.Process(
            target=worker_process,
            args=(i, gpu_id, task_queue, args, optimal_values, logs_dir, 
                  total_tasks, shared_counter, global_start_time)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_time = time.time() - global_start_time
    print("-" * 60)
    print(f"🎉 All tasks completed! Total elapsed: {format_timedelta(total_time)}")

if __name__ == "__main__":
    main()