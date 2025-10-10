PORT=30001
GPU=$1
CFG=$2
TAG=${3:-'default'}

uv run torchrun --nproc_per_node $GPU --master_port $PORT main.py --cfg $CFG --amp --wandb