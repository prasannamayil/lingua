#!/bin/bash

# Get arguments
NPROC=$1
CONFIG=$2
API_KEY=$3
EVAL_NAME=${4:-"multi_eval_$(date +%Y%m%d_%H%M%S)"}  # Default name with timestamp

# Source virtual environment
source ~/.lingua_241216/bin/activate

# Setup tokenizer
python setup/download_tokenizer.py llama3 /tmp/tokenizers/ --api_key=$API_KEY

# Read base directory from config
BASE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('apps/mamba/configs/$CONFIG'))['metric_log_dir'])")

# Create new evaluation directory
EVAL_DIR="${BASE_DIR}/${EVAL_NAME}"
mkdir -p "${EVAL_DIR}/evals"
cp "apps/mamba/configs/${CONFIG}" "${EVAL_DIR}/config.yaml"

# Find all checkpoint directories
find "${BASE_DIR}/checkpoints" -maxdepth 1 -type d -name "[0-9]*" | sort -n | while read -r ckpt_dir; do
    if [ -d "$ckpt_dir" ]; then
        # Extract the exact checkpoint number by removing leading zeros
        ckpt_num=$(basename "$ckpt_dir" | sed 's/^0*//')
        echo "Evaluating checkpoint: $ckpt_num"
        
        # Create temporary config for this checkpoint
        TMP_CONFIG="${EVAL_DIR}/tmp_${ckpt_num}.yaml"
        sed "s|ckpt_dir:.*|ckpt_dir: ${ckpt_dir}|" "apps/mamba/configs/${CONFIG}" > "$TMP_CONFIG"
        sed -i "s|dump_dir:.*|dump_dir: ${EVAL_DIR}/evals/${ckpt_num}|" "$TMP_CONFIG"
        sed -i "s|metric_log_dir:.*|metric_log_dir: ${EVAL_DIR}|" "$TMP_CONFIG"
        # Add global_step to the config
        echo "global_step: ${ckpt_num}" >> "$TMP_CONFIG"
        
        # Run evaluation
        export NCCL_IB_DISABLE=1
        torchrun --nproc_per_node=$NPROC -m apps.mamba.eval config="$TMP_CONFIG"
        
        # Cleanup temporary config
        rm "$TMP_CONFIG"
    fi
done

echo "All evaluations completed. Results are in: ${EVAL_DIR}" 