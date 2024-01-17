#!/bin/bash

#SBATCH -J VICReg-RN10
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --time=4:58:00
#SBATCH --partition=a40
#SBATCH --qos=m2
#SBATCH --export=ALL
#SBATCH --output=out_tau_cct.log
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90

export PYTHONPATH=/h/harmanan/TRUSnet
cd projects/TRUS_ViT

/h/harmanan/anaconda3/envs/trusnet/bin/python main.py experiment=ssl_resnet10 fold=4