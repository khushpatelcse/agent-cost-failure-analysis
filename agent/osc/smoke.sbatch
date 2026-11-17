#!/bin/bash
#SBATCH --account=PJS0401
#SBATCH --job-name=f1-smoke
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# Submit from the project folder:  mkdir -p logs && sbatch osc/smoke.sbatch
module load miniconda3/24.1.2-py310/24.1.2-py310
source activate f1llama
export HF_HUB_OFFLINE=1         # use the weights downloaded in setup_env.sh
nvidia-smi
python smoke_test.py
