
export CUDA_VISIBLE_DEVICES=1

python clip_feature.py \
    --image_root /scratch/zl4789/MQL4GRec/data_process/amazon18_data/Images \
    --save_root /scratch/zl4789/MQL4GRec/data_process/MQL4GRec \
    --model_cache_dir /scratch/zl4789/MQL4GRec/cache_models/clip \
    --dataset Instruments


