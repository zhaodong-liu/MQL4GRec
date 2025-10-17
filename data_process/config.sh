# data_process/config.sh
# 统一参数配置（可被临时覆盖）
: "${DATASET:=Games}"                      # 默认数据集
: "${AMAZON18_INPUT_PATH:=amazon18_data}"   # 你的下载/原始路径
: "${OUTPUT_PATH:=MQL4GRec}"        # 处理后输出根目录
: "${IMAGE_ROOT:=amazon18_data/Images}"
: "${MODEL_CACHE_DIR:=/scratch/zl4789/MQL4GRec/.cachemodels/clip}"
: "${CUDA_VISIBLE_DEVICES:=1}"                   # 需要时会被脚本使用

# 常用的目录派生（可按需添加）
SAVE_ROOT="${OUTPUT_PATH}"                       # 你脚本里用到的 save_root
