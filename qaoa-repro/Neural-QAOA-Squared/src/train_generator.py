# train_generator.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
import psutil
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
import random
import math
import numpy as np
from typing import List, Optional, Union
from utils import format_bytes

from src.models.generator import JointGenerator

from src.models.critic_r import Critic_R

from data import ActorGraphDataset


RESUME_TRAINING = False

def toggle_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def train_generator(critic_id: Optional[str] = None, generator_id: Optional[str] = None, fine_tune: bool = False):
    print("--- Training the JointGenerator ---")
    
    process = psutil.Process(os.getpid())
    total_available_mem = psutil.virtual_memory().total
    print(f"WSL environment total memory: {format_bytes(total_available_mem)}")

    print(f"Start loading GENERATOR graph dataset at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    dataset = ActorGraphDataset(config.PARTITION_GENERATOR_GRAPH_DATASET_PATH)
    print(f"Finished loading dataset at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total graph dataset size: {len(dataset)} graphs.")


    train_size = int(config.SPLIT_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    

    REAL_BATCH_SIZE = config.GENERATOR_BATCH_SIZE  
    
    worker_count = 8

    train_loader = PyGDataLoader(
        train_dataset, 
        batch_size=REAL_BATCH_SIZE, 
        shuffle=True,
        num_workers=worker_count,
        pin_memory=True
    )
    val_loader = PyGDataLoader(
        val_dataset, 
        batch_size=REAL_BATCH_SIZE, 
        shuffle=False,
        num_workers=worker_count,
        pin_memory=True
    )
    print(f"Train graphs: {len(train_dataset)} | Val graphs: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Using Physical Batch Size = {REAL_BATCH_SIZE}")

    # PRE-TRAINED Critic
    print("Loading pre-trained Critic_R model...")
    critic_model = Critic_R().to(config.DEVICE)
    
    if critic_id is None:
        critic_id = "1765159002" 

    critic_model_path = os.path.join(config.CRITIC_MODEL_PATH, f"critic_r_best_model_{critic_id}.pth")
    
    if not os.path.exists(critic_model_path):
        print(f"Error: Pre-trained Critic model not found at {critic_model_path}")
        print("Please train the Critic model first using 'train_critic.py'.")
        return
        
    critic_model.load_state_dict(torch.load(critic_model_path, map_location=config.DEVICE))
    critic_model.eval()
    for param in critic_model.parameters():
        param.requires_grad = False
    print(f"Critic_R {critic_id} loaded, set to eval mode, and parameters frozen.")

    # Generator
    model = JointGenerator().to(config.DEVICE)
    print("JointGenerator model initialized.")
    
    current_lr = config.GENERATOR_LEARNING_RATE
    
    best_val_loss = float('inf')

    best_global_loss = float('inf')

    start_epoch = 0

    global_step = 0

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=current_lr, 
        weight_decay=config.GENERATOR_WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=100)


    if generator_id is not None:
        resume_path = os.path.join(config.PARTITION_GENERATOR_MODEL_PATH, f"generator_best_model_{generator_id}.pth")
        
        if os.path.exists(resume_path):
            print(f"Loading checkpoint from {resume_path}...")
            checkpoint = torch.load(resume_path, map_location=config.DEVICE)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            if fine_tune:
                print("--- Fine-tuning Mode Activated ---")
                print("1. Model weights loaded.")
                print("2. Optimizer/Scheduler reset (starting fresh).")
                print("3. Epoch count reset to 0.")
                
                current_lr = current_lr * 0.25  
                print(f"4. Learning Rate reduced to: {current_lr}")

                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
            elif RESUME_TRAINING: 

                if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"Resumed training state from epoch {checkpoint['epoch']}")
        else:
            print(f"Warning: Model {generator_id} not found, training from scratch.")


    if fine_tune or (not RESUME_TRAINING):
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=current_lr,
            weight_decay=config.GENERATOR_WEIGHT_DECAY
        )

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=100)

    

    run_id = int(time.time())

    log_dir = os.path.join("runs", f"generator_training_{run_id}")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard log will be saved to: {log_dir}")
    print(f"Current Run ID: {run_id}") 

    save_dir = config.PARTITION_GENERATOR_MODEL_PATH
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model save path set to: {save_dir}")

    np.set_printoptions(threshold=sys.maxsize, linewidth=200, precision=4, suppress=True)

    init_save_path = os.path.join(save_dir, f"generator_best_model_{run_id}.pth")
    torch.save({
            'epoch': start_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_global_loss': best_global_loss
        }, init_save_path)
    print(f"--- Initialized model state saved to: {init_save_path} ---")

    for epoch in range(start_epoch, config.GENERATOR_NUM_EPOCHS):

        # --- Training ---
        model.train()
        total_train_loss = 0
        mem_str = ""

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.GENERATOR_NUM_EPOCHS} [Train]")

        for i, data in enumerate(train_pbar):
            data = data.to(config.DEVICE)
            optimizer.zero_grad()
   
            node_counts = torch.bincount(data.batch) 
            
            # k = ceil(N / MAX_CAP) = (N + MAX_CAP - 1) // MAX_CAP
            k_tensor = (node_counts + config.MAX_NODES_PER_PARTITION - 1) // config.MAX_NODES_PER_PARTITION
            k_tensor = k_tensor.long() # [Batch_Size]
            partition_indices, P, data = model(data, k_tensor, config.MAX_NODES_PER_PARTITION)
            ratio = critic_model(data) 
            if ratio.dim() > 1: ratio = ratio.squeeze()
            loss = -ratio.mean()

            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_train_loss += loss.item()

            if i % 100 == 0:
                used_mem = process.memory_info().rss
                available_mem = psutil.virtual_memory().available
                mem_str = f"Mem: {format_bytes(used_mem)} / {format_bytes(available_mem)}"

            train_pbar.set_postfix(
                loss=f"{loss.item():.12f}",
                avg_loss=f"{total_train_loss/(i+1):.12f}",
                mem=mem_str
            )
            
            writer.add_scalar('Loss_Step/train_total', loss.item(), global_step)
            global_step += 1
            
        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss_Epoch/train_total', avg_train_loss, epoch)

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.GENERATOR_NUM_EPOCHS} [Val]")):
                data = data.to(config.DEVICE)
                
                node_counts = torch.bincount(data.batch)
                k_tensor = (node_counts + config.MAX_NODES_PER_PARTITION - 1) // config.MAX_NODES_PER_PARTITION
                k_tensor = k_tensor.long()

                partition_indices, P, data = model(data, k_tensor, config.MAX_NODES_PER_PARTITION)
                ratio = critic_model(data)
                if ratio.dim() > 1: ratio = ratio.squeeze()
                loss = -ratio.mean()
                total_val_loss += loss.item()
                
        if len(val_loader) == 0:
            print("[Warning] Validation loader is empty, skipping validation loss calculation.")
            avg_val_loss = avg_train_loss
        else:
            avg_val_loss = total_val_loss / len(val_loader)
        
        writer.add_scalar('Loss_Epoch/val_total', avg_val_loss, epoch)
        writer.add_scalar('Hyperparameters/learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch+1}/{config.GENERATOR_NUM_EPOCHS}, Train Loss: {avg_train_loss:.12f}, Val Loss: {avg_val_loss:.12f}") 
        
        avg_global_loss = 0.5 * (avg_train_loss + avg_val_loss)

        writer.add_scalar('Loss_Epoch/global_total', avg_global_loss, epoch)

        scheduler.step(avg_global_loss)
        

        if avg_global_loss < best_global_loss:
            best_global_loss = avg_global_loss
            save_path = os.path.join(save_dir, f"generator_best_model_{run_id}.pth")
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_global_loss': best_global_loss
                }, save_path)
            print(f"Global loss {avg_global_loss:.12f} improved. Model saved to {save_path}")

    writer.close()
    print("--- Generator training finished ---")


def test_generator(critic_id: Optional[str] = None, generator_id: Optional[str] = None):

    print("\n--- Testing the JointGenerator with Detailed Output ---")
    
    if generator_id is None:
        generator_id = "1765270163" 
    model_path = os.path.join(config.PARTITION_GENERATOR_MODEL_PATH, f"generator_best_model_{generator_id}.pth")
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found at: {model_path}")
        return

    test_path = config.PARTITION_GENERATOR_TEST_GRAPH_DATASET_PATH
        
    print(f"Loading test graph dataset from: {test_path}")
    try:
        test_dataset = ActorGraphDataset(test_path)
        if len(test_dataset) == 0:
            print("[Warning] Test graph dataset is empty.")
            return
        print(f"Test dataset size: {len(test_dataset)} graphs.")
    except Exception as e:
        print(f"[Error] Failed to load test dataset: {e}")
        return

    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=1, 
        shuffle=False, 
        num_workers=0 
    )

    model = JointGenerator().to(config.DEVICE)
    
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded best Generator model from {model_path} (checkpoint format)")
        if 'epoch' in checkpoint:
            print(f"  Model was saved at epoch {checkpoint['epoch']}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Successfully loaded best Generator model from {model_path} (legacy format)")
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("JointGenerator loaded, set to eval mode, and parameters frozen.")
    
    print("Loading pre-trained Critic_R model for evaluation...")
    critic_model = Critic_R().to(config.DEVICE)
    
    if critic_id is None:
        critic_id = "1765159002"  
    
    critic_model_path = os.path.join(config.CRITIC_MODEL_PATH, f"critic_r_best_model_{critic_id}.pth")
    
    if not os.path.exists(critic_model_path):
        print(f"Error: Pre-trained Critic model not found at {critic_model_path}")
        return
    critic_model.load_state_dict(torch.load(critic_model_path, map_location=config.DEVICE))
    critic_model.eval()
    for param in critic_model.parameters():
        param.requires_grad = False
    print("Critic_R loaded, set to eval mode, and parameters frozen.")



    timestamp = int(time.time())
    output_filename = f"generator_test_results_{timestamp}.txt"
    output_path = os.path.join(config.PARTITION_GENERATOR_MODEL_PATH, output_filename)
    print(f"Results will be saved to: {output_path}")


    torch.set_printoptions(profile="full", linewidth=200, precision=4, sci_mode=False)
    np.set_printoptions(threshold=sys.maxsize, linewidth=200, precision=4, suppress=True)


    total_score = 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"Generator Test Results - {time.ctime()}\n")
        f_out.write(f"Model: {model_path}\n")
        f_out.write("="*80 + "\n\n")

        with torch.no_grad(): 
            for batch_idx, data in enumerate(tqdm(test_loader, desc="Testing Generator")):
                data = data.to(config.DEVICE)
                
                node_counts = torch.bincount(data.batch) 
                k_tensor = (node_counts + config.MAX_NODES_PER_PARTITION - 1) // config.MAX_NODES_PER_PARTITION
                k_tensor = k_tensor.long()
                
                k_val = k_tensor.item()
                N = data.num_nodes

                # partition_indices: [N]
                # P: [Total_K, 2*depth]
                # data
                partition_indices, P, data = model(data, k_tensor, config.MAX_NODES_PER_PARTITION)

                alpha_q = critic_model(data)
                if alpha_q.dim() > 1: alpha_q = alpha_q.squeeze()

                Q_pred = alpha_q * (data.OPT - data.r)
                numerator = Q_pred + data.r - data.sum_neg
                denominator = data.OPT - data.sum_neg
                
                pred_global_ratio = numerator / denominator
                current_score = pred_global_ratio.item()
                total_score += current_score
                
                partition_list = [[] for _ in range(k_val)]
                idx_list = partition_indices.cpu().tolist()
                for node_id, part_id in enumerate(idx_list):
                    partition_list[part_id].append(node_id)
                
                P_numpy = P.detach().cpu().numpy()

                f_out.write(f"--- Instance {batch_idx} ---\n")
                f_out.write(f"Nodes: {N}, Partitions (k): {k_val}, Max Capacity: {config.MAX_NODES_PER_PARTITION}\n")
                f_out.write(f"Global Ratio Score: {current_score:.6f}\n")
                
                f_out.write("Global Subgraph Parameters:\n")
                f_out.write(f"  OPT: {data.OPT.item():.4f}, r: {data.r.item():.4f}, sum_neg: {data.sum_neg.item():.4f}\n")
                f_out.write(f"  Alpha: {alpha_q.item():.4f}\n")
                
                f_out.write(f"\nPartition Details (Nodes & QAOA Parameters):\n")
                
                for p_id in range(k_val):

                    p_str = np.array2string(P_numpy[p_id], separator=', ', precision=6)
                    
                    f_out.write(f"  [Partition {p_id}]\n")
                    f_out.write(f"    QAOA Params P[{p_id}]: {p_str}\n")
                    f_out.write(f"    Nodes (Size {len(partition_list[p_id])}): {partition_list[p_id]}\n")
                
                f_out.write("-" * 80 + "\n\n")
                
                if batch_idx % 10 == 0:
                    f_out.flush()

    if len(test_loader) > 0:
        avg_test_score = total_score / len(test_loader)
        summary_msg = (
            f"\n--- Generator Test Summary ---\n"
            f"Total Graphs: {len(test_loader)}\n"
            f"Average Score: {avg_test_score:.12f}\n"
            f"Saved to: {output_path}\n"
        )
        print(summary_msg)
        with open(output_path, "a", encoding="utf-8") as f_out:
            f_out.write(summary_msg)
            
    else:
        print("[Error] Test loader was empty, cannot calculate averages.")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or test the PartitionGenerator model")
    # 'train', 'test',
    parser.add_argument("--mode", type=str, default="train", help="Mode: 'train', 'test'")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune mode: load weights but reset optimizer/epochs")
    parser.add_argument("--c", type=str, required=False, help="Load critic model")
    parser.add_argument("--g", type=str, required=False, help="Load generator model")
    args = parser.parse_args()
    
    if args.mode == "train":
        if args.resume:
            print("Resuming training from checkpoint...")
            RESUME_TRAINING = True
        train_generator(args.c, args.g, fine_tune=args.finetune)
    elif args.mode == "test":
        # This is the standard test (using torch.no_grad())
        test_generator(args.c, args.g)

    else:
        print("Invalid argument. Choose 'train', 'test'.")
