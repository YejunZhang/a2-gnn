#!/bin/bash
#SBATCH --job-name=gomatch
#SBATCH --account=project_2002051
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:v100:1


python test.py