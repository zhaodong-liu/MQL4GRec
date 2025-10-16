
export CUDA_VISIBLE_DEVICES=1

python data_process/clip_feature.py \
    --image_root /amazon18_data/Images \
    --save_root /MQL4GRec \
    --model_cache_dir MQL4GRec/cache_models/clip \
    --dataset Instruments


