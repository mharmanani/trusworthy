#!/bin/bash

#SBATCH -J run
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --time=2:58:00
#SBATCH --partition=a40,t4v2,rtx6000
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=vitus.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90

export PYTHONPATH=/h/harmanan/projects/TRUSnet
cd projects/TRUS_ViT

/h/harmanan/anaconda3/envs/trusnet/bin/python main.py experiment=vitus fold=0