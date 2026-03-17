#!/bin/bash -l
#SBATCH --job-name=mnist_mlp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=canli
#SBATCH --partition=a100-80gb
#SBATCH --output=joboutput/mnist-%j.out

set -euo pipefail

module purge
unset PYTHONPATH
export PYTHONNOUSERSITE=1
module load external
module load conda/2025.09

source activate /scratch/gilbreth/$USER/conda_envs/torch_gpu

cd "$SLURM_SUBMIT_DIR"

echo "JOBID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('cuda_built=', torch.backends.cuda.is_built()); print('torch_cuda_version=', torch.version.cuda); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

python -c "import torch, torchvision, numpy; print('torch=', torch.__version__); print('torchvision=', torchvision.__version__); print('numpy=', numpy.__version__)"

python -u train_mnist.py \
  --epochs 5 \
  --batch-size 256 \
  --num-workers 4 \
  --data-dir "$HOME/data/mnist"