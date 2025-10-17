# data_process/config.sh
# 统一参数配置（可被临时覆盖）
: "${DATASET:=Games}"
: "${AMAZON18_INPUT_PATH:=amazon18_data}"
: "${OUTPUT_PATH:=MQL4GRec}"
: "${IMAGE_ROOT:=amazon18_data/Images}"
: "${MODEL_CACHE_DIR:=/scratch/yh4663/MQL4GRec/.cachemodels/clip}"
: "${CUDA_VISIBLE_DEVICES:=1}"

SAVE_ROOT="${OUTPUT_PATH}"
