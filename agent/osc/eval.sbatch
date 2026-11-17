#!/bin/bash
#SBATCH --account=PJS0401
#SBATCH --job-name=f1-eval
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
module load miniconda3/24.1.2-py310/24.1.2-py310
source activate f1llama
export HF_HUB_OFFLINE=1

# Before vs after fine-tuning, same test set. Gemini is run on a laptop.
python evaluate.py --model llama --tag llama-base
python evaluate.py --model llama --adapter outputs/f1-lora --tag llama-lora
