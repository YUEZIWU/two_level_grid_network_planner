#!/bin/bash
#SBATCH --job-name=TLND # Job name
#SBATCH --mail-type=END,FAIL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=amisiko@umass.edu
#SBATCH --partition=cpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 6 # one processor
#SBATCH --mem=30G  # Requested Memory
#SBATCH -t 01:00:00  # Job time limit
#SBATCH --output=TLND_%A.log

source ~/uganda_network/network_design_env/bin/activate
python unity_TwoLevelNetworkDesign.py
