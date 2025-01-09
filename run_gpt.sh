# Get arguments
NPROC=$1
CONFIG=$2
API_KEY=$3
source ~/.lingua_241216/bin/activate
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not needed
export NCCL_P2P_DISABLE=1  # Disable P2P if causing issues
export CUDA_LAUNCH_BLOCKING=1 # for debugging
torchrun --nproc-per-node $NPROC -m apps.gpt2.train config=apps/gpt2/configs/$CONFIG
