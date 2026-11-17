#!/bin/bash
#SBATCH --account=PJS0401
#SBATCH --job-name=f1-train
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# mkdir -p logs && sbatch osc/train.sbatch     (check with: squeue -u $USER)
module load miniconda3/24.1.2-py310/24.1.2-py310
source activate f1llama
export HF_HUB_OFFLINE=1

TRAIN="data/plan_train.jsonl"
[ -f data/answer_train.jsonl ] && TRAIN="$TRAIN data/answer_train.jsonl"

python train_lora.py --train $TRAIN --val data/plan_val.jsonl --out outputs/f1-lora
