#!/bin/bash

# define all candidate policies
POLICIES=(
    "modularity"
    "random"
    "kl"
    "boundary"
    "JointGenerator+Critic"
)

# define script and data paths (using variables for easy modification)
SCRIPT_PATH="scripts/batch-process-parallel.py"
# ALGO_PATH="data/data_generators/QAOA_in_QAOA.py"
ALGO_PATH="competitors/QAOA-in-QAOA/QAOA_in_QAOA.py"
DATA_PATH="data/instances/data/test_instances_only/mc"
OPTIMAL_FILE="data/instances/data/osv.json"

# Master process bound CPU core ID
MASTER_CPU=9

echo "Starting sequential batch processing on Linux..."
echo "Master process will be bound to CPU core: ${MASTER_CPU}"

# Loop through policies sequentially
for policy in "${POLICIES[@]}"; do
    echo ""
    echo "----------------------------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Policy: ${policy}"
    echo "----------------------------------------------------------------"

    # Execute command
    # 1. taskset -c 9: Ensure the python main process only occupies CPU 9
    # 2. This command is blocking, it must finish before the next loop iteration
    taskset -c ${MASTER_CPU} python "${SCRIPT_PATH}" \
      --algorithm "${ALGO_PATH}" \
      --dataset_path "${DATA_PATH}" \
      --experiment m \
      --policy "${policy}" \
      --base qaoa \
      --depth 1 \
      --optimal_values_file "${OPTIMAL_FILE}" \
      --runs 10 \
      --gpus 0,1,2,3

    # Capture exit status
    EXIT_CODE=$?
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "✅ Policy '${policy}' executed successfully."
    else
        echo "❌ Policy '${policy}' failed (Exit Code: ${EXIT_CODE})."
        # If you want to stop all subsequent tasks immediately upon error, uncomment the following line:
        # exit 1
    fi

    # Optional: simple cooldown time to allow system stdout buffer to flush
    sleep 2
done

echo ""
echo "🎉 All policy tasks have been completed!"