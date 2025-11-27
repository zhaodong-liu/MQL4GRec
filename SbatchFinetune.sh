#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --account=pr_119_tandon_priority
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=yh4663@nyu.edu
#SBATCH --requeue

source ~/.bashrc
conda activate MQL-2.9.0
cd /scratch/yh4663/MQL4GRec
bash scripts/finetune.sh
conda deactivate