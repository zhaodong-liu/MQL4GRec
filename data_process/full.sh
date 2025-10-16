#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --account=pr_119_tandon_priority
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=zl4789@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate multimodal
cd /scratch/zl4789/MQL4GRec


bash /scratch/zl4789/MQL4GRec/data_process/0_download.sh
bash /scratch/zl4789/MQL4GRec/data_process/1_load_figure.sh
bash /scratch/zl4789/MQL4GRec/data_process/2_process.sh
# bash /scratch/zl4789/MQL4GRec/data_process/3_get_text_emb.sh
bash /scratch/zl4789/MQL4GRec/data_process/4_get_image_emb.sh

conda deactivate