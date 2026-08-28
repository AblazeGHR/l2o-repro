# config.py
import os
from pathlib import Path
import torch
import random
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Random seed (optional) ---
# Set to an integer to make experiments deterministic, or None to leave randomness.
SEED = 42
# Python built-in RNG
random.seed(SEED)
# NumPy RNG
np.random.seed(SEED)
# PyTorch CPU/GPU RNG
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# For reproducibility you can fix cuDNN behavior; it may slightly reduce performance.
# Recommended for research/debugging; consider disabling in production if you prefer performance.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
# --- PennyLane / quantum RNG compatibility ---
# Ensure pennylane.numpy and PennyLane RNGs are also seeded when available.
# This makes experiments more reproducible when competitors use pennylane.numpy.
# Set pennylane.numpy RNG
pnp.random.seed(SEED)
# Some PennyLane backends expose a helper to set RNG seed globally
# qml.set_rng_seed was introduced in some versions; try to call it if present.
if hasattr(qml, 'set_rng_seed'):
    qml.set_rng_seed(SEED)

# --- Model Hyperparameters ---
MAX_NODES_PER_PARTITION = 10      # Max nodes per subgraph (physical constraint)
NODE_FEATURE_DIM = 5              # Node feature dim (degree + weighted degree + clustering coeff + PageRank + betweenness centrality)
EDGE_FEATURE_DIM = 1              # Edge feature dim (edge weight)
GNN_HIDDEN_DIM = 64               # Hidden dim of GNN intermediate layers
GNN_NUM_LAYERS = 3                # Number of GNN layers
MLP_HIDDEN_DIM = 256              # Hidden dim of the prediction-head MLP

# --- Training Hyperparameters ---
# Critic
TRAINING_SET_DIR = "result/preliminary"
TESTING_SET_DIR = "result/preliminary"
CRITIC_LEARNING_RATE = 1e-3
CRITIC_WEIGHT_DECAY = 5e-4
CRITIC_NUM_EPOCHS = 100
CRITIC_BATCH_SIZE = 32
CRITIC_DATASET_PATH = "data/datasets/training-set/critic_dataset16.pkl"
CRITIC_TEST_DATASET_PATH = "data/datasets/testing-set/critic_test_dataset16.pkl"
# CRITIC_MODEL_PATH = "checkpoints/critic_r/"
# 1. Extract a unique identifier from the dataset path (e.g., "4")
try:
    # "critic_dataset4.pkl" -> "critic_dataset4"
    dataset_stem = Path(CRITIC_DATASET_PATH).stem
    # "critic_dataset4" -> "4"
    dataset_version = dataset_stem.split('dataset')[-1]
    if not dataset_version: # Just in case the filename is "critic_dataset.pkl"
        dataset_version = "N/A"
except Exception:
    dataset_version = "unknown"

# 2. Format key training parameters
lr_str = f"{CRITIC_LEARNING_RATE:.0e}"
wd_str = f"{CRITIC_WEIGHT_DECAY:.0e}"

# 3. Compose a descriptive, filesystem-safe folder name
model_name = (
    f"Critic_R_Data{dataset_version}"
    f"_GNN-L{GNN_NUM_LAYERS}-H{GNN_HIDDEN_DIM}"
    f"_MLP-H{MLP_HIDDEN_DIM}"
    f"_NF{NODE_FEATURE_DIM}" 
    f"_NE{CRITIC_NUM_EPOCHS}"
    f"_BS{CRITIC_BATCH_SIZE}"
    f"_LR{lr_str}"
    f"_WD{wd_str}"
)

# 4. Dynamically set the final model save path
CRITIC_MODEL_PATH = f"checkpoints/critic_r/{model_name}/"


# print(f"Model save path set to: {CRITIC_MODEL_PATH}")
# os.makedirs(CRITIC_MODEL_PATH, exist_ok=True)

ACTOR_GRAPH_DATASET_PATH = "data/datasets/training-set/graph_dataset5.pkl"
ACTOR_TEST_GRAPH_DATASET_PATH = "data/datasets/testing-set/graph_test_dataset5.pkl"

# Local Search
LOCAL_SEARCH_BATCH_SIZE = 1


# Simulated Annealing

# N-tournament size: sample N neighbors per iteration
SA_TOURNAMENT_SIZE = 100 

# Total number of iterations
SA_MAX_ITERATIONS = 2000 

# Initial temperature (accept a solution worse by 0.01 with ~80% probability)
SA_INITIAL_TEMPERATURE = 0.005 # -0.01 / math.log(0.8)

# Minimum temperature (stopping criterion)
SA_MIN_TEMPERATURE = 1e-6

# Cooling rate
SA_COOLING_RATE = 0.999



# PartitionGenerator
PARTITION_GENERATOR_GRAPH_DATASET_PATH = "data/datasets/training-set/graph_dataset5.pkl"
PARTITION_GENERATOR_TEST_GRAPH_DATASET_PATH = "data/datasets/testing-set/graph_test_dataset5.pkl"
PARTITION_GENERATOR_MODEL_PATH = "checkpoints/partition_generator/"
GENERATOR_GNN_HIDDEN_DIM = 128               # Hidden dim of GNN intermediate layers
GENERATOR_GNN_NUM_LAYERS = 2                 # Number of GNN layers
GENERATOR_MLP_HIDDEN_DIM = 128               # Hidden dim of MLP layers
EMBED_DIM = 128                              # Embedding dim projected to the clustering space (input to OrthogonalComplementHead)

CLUSTER_HEAD_TEMP = 0.05 # Softmax temperature $\tau$ in OrthogonalComplementHead

 
GENERATOR_NUM_EPOCHS = 1500
GENERATOR_BATCH_SIZE = 16
GENERATOR_LEARNING_RATE = 4e-3
GENERATOR_WEIGHT_DECAY = 5e-4
# ================================================================================================

INFERENCE_LEARNING_RATE = 1e-3

# QAOA
QAOA_DEPTH = 1                  # QAOA circuit depth p

PARTITION_ONLY_EPOCHS_RATIO = 0.5  # Fraction of total epochs to pretrain the partition generator
SPLIT_RATIO = 0.9                  # Train/validation split ratio