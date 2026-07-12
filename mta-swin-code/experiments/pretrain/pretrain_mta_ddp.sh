#!/bin/bash
#SBATCH --job-name=mta-tiny-ddp
#SBATCH -p multigpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=outputs/slurm_logs/pretrain/train_%j.out
#SBATCH --error=outputs/slurm_logs/pretrain/train_%j.err

# Basic info
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"

# Create output directories
mkdir -p outputs/slurm_logs/pretrain

# Check GPU
nvidia-smi

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate dl_brain

# PyTorch CUDA check
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Set environment variables for multi-GPU training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355
export WORLD_SIZE=4
export NCCL_DEBUG=INFO

# Run multi-GPU training using torchrun
# 4 GPU training
torchrun --nproc_per_node=4 \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         experiments/pretrain/pretrain_mta_ddp.py \
         --config experiments/pretrain/configs/mta_swin_ddp_v2.yaml \
        #  --resume outputs/checkpoints/pretrain/mta_swin_tiny_20250915_011146/latest_model.pth