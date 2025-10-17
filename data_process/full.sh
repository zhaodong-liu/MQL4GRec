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

DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/config.sh"

DATASETS=(CDs Instruments Sports Games Beauty Arts)

# config.sh now provides absolute paths, so no need to convert
# But if user overrides with relative paths, convert them
[[ "$AMAZON18_INPUT_PATH" != /* ]] && AMAZON18_INPUT_PATH="$PROJECT_ROOT/${AMAZON18_INPUT_PATH#./}"
[[ "$OUTPUT_PATH" != /* ]] && OUTPUT_PATH="$PROJECT_ROOT/${OUTPUT_PATH#./}"
[[ "$IMAGE_ROOT" != /* ]] && IMAGE_ROOT="$PROJECT_ROOT/${IMAGE_ROOT#./}"

mkdir -p "$AMAZON18_INPUT_PATH"
mkdir -p "$OUTPUT_PATH"

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate multimodal

# Get the absolute path of MQL4GRec directory (parent of data_process)
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES

for DATASET in "${DATASETS[@]}"; do
  python "$DIR/0_download_amazon_data.py" \
    --dataset "$DATASET" \
    --output_path "$AMAZON18_INPUT_PATH"

  python "$DIR/load_all_figures.py" --dataset "$DATASET"

  python "$DIR/amazon18_data_process.py" \
    --dataset "$DATASET" \
    --input_path "$AMAZON18_INPUT_PATH" \
    --output_path "$OUTPUT_PATH"

  python "$DIR/amazon_text_emb.py" --dataset "$DATASET"

  python "$DIR/clip_feature.py" \
    --image_root "$IMAGE_ROOT" \
    --save_root "$SAVE_ROOT" \
    --model_cache_dir "$MODEL_CACHE_DIR" \
    --dataset "$DATASET"
done

conda deactivate