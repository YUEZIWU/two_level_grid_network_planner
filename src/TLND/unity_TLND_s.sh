#!/bin/bash
#SBATCH --job-name=TLND # Job name
#SBATCH --mail-type=END,FAIL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yw3054@columbia.edu
#SBATCH --partition=cpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 6 # one processor
#SBATCH --mem=30G  # Requested Memory
#SBATCH -t 01:00:00  # Job time limit
#SBATCH --output=TLND_%A.log

conda activate base
conda activate tlnd
python TwoLevelNetworkDesign.py
