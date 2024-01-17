#!/bin/bash

#SBATCH -J RN18l
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --qos=m3
#SBATCH --time=3:59:00
#SBATCH --partition=a40,t4v2,rtx6000
#SBATCH --export=ALL
#SBATCH --output=out2.log
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90

export PYTHONPATH=/h/harmanan/TRUSnet
cd projects/TRUS_ViT

# ALL: [42, 142, 292, 392, 126, 178, 81, 512, 659, 5, 602, 321]
# DONE: 292 5 81 512 126 602 178 392 42 659 321 881 452
# REMAINING: 148

#centerwise list: 42x 142x 5x 81e 1x 126 292x 321 392x 452 512 602 659 881

/h/harmanan/anaconda3/envs/trusnet/bin/python main.py experiment=isbi_resnet10 seed=321 fold=4