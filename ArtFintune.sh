#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --nodes=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --account=pr_119_tandon_priority
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=yh4663@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2024.02.07/etc/profile.d/conda.sh;
conda activate MQL
cd /scratch/yh4663/MQL4GRec
bash scripts/finetune.sh
conda deactivate