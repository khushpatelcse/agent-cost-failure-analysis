#!/bin/bash
# Run ONCE on an Ascend login node (ssh ascend.osc.edu, or OnDemand > Clusters > Ascend Shell).
# Replace PJS0401 with your class project code everywhere in this folder.
set -e
PROJECT=PJS0401

# Big files (model weights, HF cache) go in scratch, not your small $HOME.
export HF_HOME=/fs/scratch/$PROJECT/$USER/hf_cache
mkdir -p "$HF_HOME"
grep -q "HF_HOME" ~/.bashrc || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

module load miniconda3/24.1.2-py310/24.1.2-py310          # if this fails: module spider miniconda3
conda create -y -n f1llama python=3.11
source activate f1llama
pip install torch transformers peft trl datasets accelerate requests python-dotenv

# Llama is gated: accept Meta's license on the model page first, then:
hf auth login                   # paste your Hugging Face access token
hf download meta-llama/Llama-3.2-3B-Instruct
