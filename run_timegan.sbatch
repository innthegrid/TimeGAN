#!/bin/bash -l
#SBATCH --job-name=timegan_air
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=canli
#SBATCH --partition=a100-80gb
#SBATCH --output=logs/timegan-%j.out

set -euo pipefail

module purge
unset PYTHONPATH
export PYTHONNOUSERSITE=1

module load external
module load conda/2025.09

source activate /scratch/gilbreth/$USER/conda_envs/timegan_env

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

echo "DATE=$(date)"
echo "JOBID=$SLURM_JOB_ID"
nvidia-smi

python -u main_timegan.py \
  --data_name air_healthy \
  --seq_len 24 \
  --module gru \
  --hidden_dim 24 \
  --num_layer 3 \
  --iteration 10000 \
  --batch_size 128 \
  --metric_iteration 5