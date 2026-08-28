# train_critic.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import os
import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data
from tqdm import tqdm
import psutil
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter_add
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from utils import format_bytes

from src.models import Critic_R
from data import CriticDataset

def train_critic(pretrained_id=None):
    print("--- Phase 1: Training the Critic ---")
    
    process = psutil.Process(os.getpid())
    total_available_mem = psutil.virtual_memory().total
    print(f"WSL environment total memory: {format_bytes(total_available_mem)}")

    print(f"Start loading dataset at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    dataset = CriticDataset(config.CRITIC_DATASET_PATH)

    print(f"Finished loading dataset at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total dataset size: {len(dataset)} data points.")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    num_cpu_cores = os.cpu_count()
    print(f"Number of CPU cores available: {num_cpu_cores}")
    # worker_count = min(num_cpu_cores, 8) if num_cpu_cores is not None else 4
    worker_count = 8  
    print(f"Using {worker_count} workers for DataLoader")
   
    train_loader = PyGDataLoader(
        train_dataset, 
        batch_size=config.CRITIC_BATCH_SIZE, 
        shuffle=True,
        num_workers=worker_count,
        pin_memory=True
    )

    val_loader = PyGDataLoader(
        val_dataset, 
        batch_size=config.CRITIC_BATCH_SIZE,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=True
    )
    print(f"Train dataset size: {len(train_loader)}, Validation dataset size: {len(val_loader)}")

    model = Critic_R().to(config.DEVICE)

    if pretrained_id:
        pretrained_path = os.path.join(config.CRITIC_MODEL_PATH, f'critic_r_best_model_{pretrained_id}.pth')
        if os.path.exists(pretrained_path):
            print(f"--- Fine-tuning Mode ---")
            print(f"Loading pretrained weights from: {pretrained_path}")
            model.load_state_dict(torch.load(pretrained_path, map_location=config.DEVICE))
        else:
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
    else:
        print("--- Training from Scratch ---")


    current_lr = config.CRITIC_LEARNING_RATE
    if pretrained_id:
        current_lr = current_lr * 0.25  
        print(f"Adjusting Learning Rate for fine-tuning: {current_lr}")

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=current_lr, 
        weight_decay=config.CRITIC_WEIGHT_DECAY 
    )

    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    global_step = 0 

    run_id = int(time.time())
    
    # TensorBoard SummaryWriter
    log_dir = os.path.join("runs", f"critic_training_{run_id}")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard log will be saved to: {log_dir}")
    print(f"Current Run ID: {run_id}") 

    print(f"Model save path set to: {config.CRITIC_MODEL_PATH}")
    os.makedirs(config.CRITIC_MODEL_PATH, exist_ok=True)


    for epoch in range(config.CRITIC_NUM_EPOCHS):
        # --- Training ---
        model.train()
        total_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.CRITIC_NUM_EPOCHS} [Train]")
        for i, data in enumerate(train_pbar):
            data = data.to(config.DEVICE)
            optimizer.zero_grad()

            # ============== Compute loss directly from ratio ==============
            
            ratio = model(data).squeeze()

            loss = criterion(ratio, data.y.float())
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()


            if i % 1000 == 0:
                used_mem = process.memory_info().rss
                available_mem = psutil.virtual_memory().available
                mem_str = (f"Mem (Used/Avail): {format_bytes(used_mem)} / "
                        f"{format_bytes(available_mem)}")


            train_pbar.set_postfix(
                avg_train_loss=f"{total_loss/(i+1):.12f}",
                mem=mem_str
            )


            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            
            total_loss += loss.item()
            global_step += 1 
            
        avg_train_loss = total_loss / len(train_loader)
        
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.CRITIC_NUM_EPOCHS} [Val]"):
                data = data.to(config.DEVICE)

                # ============== Compute loss directly from ratio ==============

                
                ratio = model(data).squeeze()
                loss = criterion(ratio, data.y.float())

                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        

        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)

        writer.add_scalar('Hyperparameters/learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch+1}/{config.CRITIC_NUM_EPOCHS}, Train Loss: {avg_train_loss:.12f}, Val Loss: {avg_val_loss:.12f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            save_filename = f"critic_r_best_model_{run_id}.pth"
            save_path = os.path.join(config.CRITIC_MODEL_PATH, save_filename)
            torch.save(model.state_dict(), save_path)
            print(f"Val loss improved. Model saved to {save_path}")

    writer.close()
    print("--- Training finished ---")


def test_critic(id=None):
    print("\n--- Phase 2: Testing the Critic ---")
    
    if id is None:
        print("Please give A model ID!")
        return
 
    model_path = os.path.join(config.CRITIC_MODEL_PATH, f'critic_r_best_model_{id}.pth')
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found at: {model_path}")
        print("Please run the training phase first ('train_critic()').")
        return


    if not hasattr(config, 'CRITIC_TEST_DATASET_PATH') or not config.CRITIC_TEST_DATASET_PATH:
        print("[Error] `config.CRITIC_TEST_DATASET_PATH` is not defined or is empty.")
        print("Please specify the path to the test dataset in your config file.")
        return
        
    print(f"Loading test dataset from: {config.CRITIC_TEST_DATASET_PATH}")
    try:
        test_dataset = CriticDataset(config.CRITIC_TEST_DATASET_PATH)
        if len(test_dataset) == 0:
            print("[Warning] Test dataset is empty.")
            return
        print(f"Test dataset size: {len(test_dataset)} data points.")
    except Exception as e:
        print(f"[Error] Failed to load test dataset: {e}")
        return


    num_cpu_cores = os.cpu_count()
    # worker_count = min(num_cpu_cores, 8) if num_cpu_cores is not None else 4
    worker_count = 2 
    
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=config.CRITIC_BATCH_SIZE, 
        shuffle=False, 
        num_workers=worker_count,
        pin_memory=True
    )
    print(f"Using {worker_count} workers for Test DataLoader.")

    model = Critic_R().to(config.DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    except Exception as e:
        print(f"[Error] Failed to load model state_dict: {e}")
        print("This might be due to a model architecture mismatch or a corrupt file.")
        return
        
    model.eval() 
    print(f"Successfully loaded best model from {model_path}")

    criterion = torch.nn.MSELoss()

    total_test_loss = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad(): 
        for data in tqdm(test_loader, desc="Testing Model"):
            data = data.to(config.DEVICE)

            # ============== Compute loss directly from ratio ==============

            
            ratio = model(data).squeeze()
            loss = criterion(ratio, data.y.float())


            total_test_loss += loss.item()

            all_predictions.append(ratio.cpu())
            all_labels.append(data.y.float().cpu())



    if all_predictions and all_labels:

        final_predictions = torch.cat(all_predictions, dim=0).numpy()
        final_labels = torch.cat(all_labels, dim=0).numpy()
        
        results_df = pd.DataFrame({
            'True_Value': final_labels,
            'Predicted_Value': final_predictions
        })
        
        output_csv_path = os.path.join(config.CRITIC_MODEL_PATH, "critic_test_results.csv")
        
        try:
            results_df.to_csv(output_csv_path, index=False, float_format='%.12f')
            print(f"\nTest results (predictions and labels) saved to: {output_csv_path}")
        except Exception as e:
            print(f"\n[Error] Failed to save test results to CSV: {e}")
    else:
        print("\n[Warning] No predictions were generated, CSV file not created.")

    if len(test_loader) > 0:
        avg_test_mse = total_test_loss / len(test_loader)
        print("\n--- Test Results ---")
        print(f"Final Mean Squared Error (MSE) on Test Set: {avg_test_mse:.12f}")
        print("----------------------")
    else:
        print("[Error] Test loader was empty, cannot calculate MSE.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train or test the Critic model")
    parser.add_argument("--model", type=str, default="train", help="Model type: 'train' or 'test'")
    parser.add_argument("--pretrained_id", type=str, default=None, help="ID of the pretrained model to fine-tune")
    parser.add_argument("--id", type=str, required=False) 
    args = parser.parse_args()
    if args.model == "train":
        train_critic(pretrained_id=args.pretrained_id)
    elif args.model == "test":
        test_critic(args.id)