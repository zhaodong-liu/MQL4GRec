
export CUDA_VISIBLE_DEVICES=1

python data_process/clip_feature.py \
    --image_root data_process/amazon18_data/Images \
    --save_root data_process/MQL4GRec \
    --model_cache_dir cache_models/clip \
    --dataset Instruments

