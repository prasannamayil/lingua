# Get arguments
NPROC=$1
API_KEY=$2
shift 2

LINGUA_LATEST=$(ls -at ~/ | grep .lingua_ | head -n 1)
echo $LINGUA_LATEST
source ~/$LINGUA_LATEST/bin/activate
python setup/download_tokenizer.py llama3 /tmp/tokenizers/ --api_key=$API_KEY
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not needed
# export NCCL_P2P_DISABLE=1  # Disable P2P if causing issues

# Check if CONFIG contains "mamba"
if [[ "$CONFIG" == *"mamba"* ]]; then
    # Execute the mamba-specific command
    torchrun --nproc_per_node=$NPROC -m apps.mamba.eval config=apps/mamba/configs/eval.yaml "$@"
else
    # Execute the original command
    torchrun --nproc_per_node=$NPROC -m apps.main.eval config=apps/main/configs/eval.yaml "$@"
fi
