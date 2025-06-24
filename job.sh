#!/bin/bash
#SBATCH --job-name=meu_job
#SBATCH --output=saida.log
#SBATCH --error=erro.log
#SBATCH --time=00:15:00
#SBATCH --account=rrg-lelis
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

source $HOME/VQVAE/bin/activate



python train.py