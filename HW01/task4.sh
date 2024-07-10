#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --cpus-per-task=2
#SBATCH --job-name=FirstSlurm
#SBATCH --output=FirstSlurm.out
#SBATCH --error=FirstSlurm.err
#SBATCH --time=0-00:01:00

hostname