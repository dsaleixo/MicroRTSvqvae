#!/bin/bash
#SBATCH --job-name=meu_job
#SBATCH --output=saida.log
#SBATCH --error=erro.log
#SBATCH --time=6:15:00
#SBATCH --account=aip-lelis
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G 
#SBATCH --cpus-per-task=1

source $HOME/VQVAE/bin/activate





python train.py