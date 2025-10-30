

# DATASETS=(CDs Sports Games Beauty Arts Instruments)
DATASETS=(Instruments)

export CUDA_VISIBLE_DEVICES

for DATASET in "${DATASETS[@]}"; do
  python data_process/0_download_amazon_data.py \
    --dataset "$DATASET" \
    --output_path "data_process/amazon18_data"

  python data_process/load_all_figures.py --dataset "$DATASET"

  python data_process/amazon18_data_process.py \
    --dataset "$DATASET" \
    --input_path data_process/amazon18_data \
    --output_path data_process/MQL4GRec 


  python data_process/amazon_text_emb.py --dataset "$DATASET"

  python data_process/clip_feature.py \
    --image_root data_process/amazon18_data/Images \
    --save_root data_process/MQL4GRec \
    --model_cache_dir cache_models/clip \
    --dataset "$DATASET"
done
