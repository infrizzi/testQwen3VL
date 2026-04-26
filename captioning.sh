#!/bin/bash
#SBATCH --job-name=captioning
#SBATCH --output=logs/tv/capt_%j.out
#SBATCH --error=logs/tv/capt_%j.err
#SBATCH --account=tesi_lpaladino
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G

# Caricamento moduli
module unload python/3.11.11-gcc-11.4.0
module load anaconda3/2023.09-0-none-none
module load cuda/12.6.3-none-none
module load ffmpeg/7.1-gcc-11.4.0

# Attivazione environment
source activate flash_test
export VIDEO_NAME=${VIDEO_NAME:-"0IdYJGBmguM"}
export VIDEO_SUBTYPE=${VIDEO_SUBTYPE:-"Video_tv"}

# Cartelle di lavoro
cd /homes/lpaladino/testQwen3VL
export PYTHONPATH=$PWD:$PYTHONPATH

# Ottimizzazione memoria Pytorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QWEN_VL_VIDEO_READER_BACKEND=decord

# Creazione cartella logs se non presente
mkdir -p logs

# ------------------------------
# Esecuzione Inferenza
# ------------------------------
python captioning.py
