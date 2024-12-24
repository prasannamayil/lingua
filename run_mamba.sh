# Get arguments
NPROC=$1
CONFIG=$2
source ~/.lingua_241216/bin/activate
python setup/download_tokenizer.py llama3 /tmp/tokenizers/ --api_key=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
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not needed
export NCCL_P2P_DISABLE=1  # Disable P2P if causing issues
torchrun --nproc-per-node $NPROC -m apps.mamba.train config=apps/mamba/configs/$CONFIG