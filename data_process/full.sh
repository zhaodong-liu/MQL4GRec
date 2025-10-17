#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=64GB
#SBATCH --account=pr_119_tandon_priority
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zl4789@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate multimodal
cd /scratch/zl4789/MQL4GRec

set -euo pipefail

DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/config.sh"

DATASETS=(CDs Sports Games Beauty Arts)

[[ "$AMAZON18_INPUT_PATH" != /* ]] && AMAZON18_INPUT_PATH="$DIR/${AMAZON18_INPUT_PATH#./}"
[[ "$OUTPUT_PATH" != /* ]] && OUTPUT_PATH="$DIR/${OUTPUT_PATH#./}"

mkdir -p "$AMAZON18_INPUT_PATH"
mkdir -p "$OUTPUT_PATH"



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

# amazon18_dataset2fullname = {
#     'Beauty': 'All_Beauty',
#     'Fashion': 'AMAZON_FASHION',
#     'Appliances': 'Appliances',
#     'Arts': 'Arts_Crafts_and_Sewing',
#     'Automotive': 'Automotive',
#     'Books': 'Books',
#     'CDs': 'CDs_and_Vinyl',
#     'Cell': 'Cell_Phones_and_Accessories',
#     'Clothing': 'Clothing_Shoes_and_Jewelry',
#     'Music': 'Digital_Music',
#     'Electronics': 'Electronics',
#     'Gift': 'Gift_Cards',
#     'Food': 'Grocery_and_Gourmet_Food',
#     'Home': 'Home_and_Kitchen',
#     'Scientific': 'Industrial_and_Scientific',
#     'Kindle': 'Kindle_Store',
#     'Luxury': 'Luxury_Beauty',
#     'Magazine': 'Magazine_Subscriptions',
#     'Movies': 'Movies_and_TV',
#     'Instruments': 'Musical_Instruments',
#     'Office': 'Office_Products',
#     'Garden': 'Patio_Lawn_and_Garden',
#     'Pet': 'Pet_Supplies',
#     'Pantry': 'Prime_Pantry',
#     'Software': 'Software',
#     'Sports': 'Sports_and_Outdoors',
#     'Tools': 'Tools_and_Home_Improvement',
#     'Toys': 'Toys_and_Games',
#     'Games': 'Video_Games'
# }
