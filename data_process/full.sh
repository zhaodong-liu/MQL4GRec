#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --account=pr_119_tandon_priority
#SBATCH --time=15:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zl4789@nyu.edu
#SBATCH --requeue

set -euo pipefail

source data_process/config.sh

DATASETS=(CDs Instruments Sports Games Beauty Arts)

mkdir -p "$AMAZON18_INPUT_PATH"
mkdir -p "$OUTPUT_PATH"

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate multimodal

export CUDA_VISIBLE_DEVICES

for DATASET in "${DATASETS[@]}"; do
  python data_process/0_download_amazon_data.py \
    --dataset "$DATASET" \
    --output_path "$AMAZON18_INPUT_PATH"

  python data_process/load_all_figures.py --dataset "$DATASET"

  python data_process/amazon18_data_process.py \
    --dataset "$DATASET" \
    --input_path "$AMAZON18_INPUT_PATH" \
    --output_path "$OUTPUT_PATH"

  python data_process/amazon_text_emb.py --dataset "$DATASET"

  python data_process/clip_feature.py \
    --image_root "$IMAGE_ROOT" \
    --save_root "$SAVE_ROOT" \
    --model_cache_dir "$MODEL_CACHE_DIR" \
    --dataset "$DATASET"
done

conda deactivate