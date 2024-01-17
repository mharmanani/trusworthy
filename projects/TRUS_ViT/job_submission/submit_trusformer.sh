#!/bin/bash

#SBATCH -J TRUSformer
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --time=3:59:00
#SBATCH --qos=m3
#SBATCH --partition=a40,t4v2,rtx6000
#SBATCH --export=ALL
#SBATCH --output=mil_vit.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90

export PYTHONPATH=/h/harmanan/TRUSnet
cd projects/TRUS_ViT

# TRUSWorthy seeds: [5, 81, 126, 602, 42, 881, 659, 321, 292, 512]
# ALL: [42, 142, 292, 392, 126, 178, 81, 512, 659, 5, 602, 321]
# DONE: 292 5 81 512 126 602 178 392 42 659 321 881 452
# REMAINING: 148

# post_ipcai note - done:5,81,126,42,881 next up is 6032 659+

/h/harmanan/anaconda3/envs/trusnet/bin/python main.py experiment=isbi_trusformer seed=512 fold=4