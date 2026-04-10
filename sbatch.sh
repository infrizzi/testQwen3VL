#!/bin/bash
#SBATCH --job-name=testQwen3VL
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
#SBATCH --account=tesi_lpaladino
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G

# Caricamento moduli
module load anaconda3/2023.09-0-none-none
module load cuda/12.6.3-none-none

# Attivazione environment
source activate flash_test

# Cartelle di lavoro
cd /homes/lpaladino/testQwen3VL
export PYTHONPATH=$PWD:$PYTHONPATH

# Ottimizzazione memoria Pytorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Creazione cartella logs se non presente
mkdir -p logs

# ------------------------------
# Esecuzione Inferenza
# ------------------------------
python test.py