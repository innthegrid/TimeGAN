#!/bin/bash -l
#SBATCH --job-name=fcvae_gilbreth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=canli
#SBATCH --partition=a100-80gb
#SBATCH --output=joboutput/fcvae-%j.out

set -euo pipefail

# 1. Load the same environment used for MNIST
module purge
unset PYTHONPATH
export PYTHONNOUSERSITE=1
module load external
module load conda/2025.09

source activate /scratch/gilbreth/$USER/conda_envs/torch_gpu

cd "$SLURM_SUBMIT_DIR"

# 2. Check GPU status
nvidia-smi

# 3. Run FCVAE with your standard parameters. 
# We set --gpu 0 to use the allocated A100 GPU.
python -u train.py \
  --data_dir ./data/Yahoo \
  --window 48 \
  --condition_emb_dim 64 \
  --condition_mode 2 \
  --save_file ./result/Score.txt \
  --gpu 0 \
  --kernel_size 24 \
  --stride 8 \
  --dropout_rate 0.05 \
  --batch_size 512 \
  --max_epoch 30 \
  --num_workers 8