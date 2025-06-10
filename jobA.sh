#!/bin/bash
#SBATCH --time=0-00:30      # time (DD-HH:MM)

#SBATCH --mem-per-cpu=6G

#SBATCH --account=def-lelis
#SBATCH --gres=gpu:1
#SBATCH --output=meu_job_%j.out
#SBATCH --error=meu_job_%j.err
source ~/envs/VQVAE/bin/activate





python train.py
