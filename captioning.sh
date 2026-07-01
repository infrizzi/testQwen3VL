#!/bin/bash
#SBATCH --job-name=captioning
#SBATCH --output=logs/capt_%j.out
#SBATCH --error=logs/capt_%j.err
#SBATCH --account=tesi_lpaladino
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G|gpu_RTX6000_24G

# Caricamento moduli
module load ffmpeg/8.1
module load anaconda3/2023.09-0

# Loading environment
source activate flash_test

export VIDEO_NAME=${VIDEO_NAME:-"2001_A_Space_Odyssey"} # default video names
export SEGMENT_TIME=${SEGMENT_TIME:-30} # default segment time in seconds
export OVERLAP_TIME=${OVERLAP_TIME:-0} # default overlap time in seconds

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
