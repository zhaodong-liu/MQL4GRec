# data_process/config.sh
# 统一参数配置（可被临时覆盖）
: "${DATASET:=Games}"                      # 默认数据集

# Get the directory containing this config file (data_process/)
CONFIG_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# Get the parent directory (MQL4GRec/)
PROJECT_ROOT="$(dirname "$CONFIG_DIR")"

# 使用相对于项目根目录的路径，如果用户没有设置环境变量
: "${AMAZON18_INPUT_PATH:=${PROJECT_ROOT}/amazon18_data}"   # 下载/原始路径
: "${OUTPUT_PATH:=${PROJECT_ROOT}/MQL4GRec}"        # 处理后输出根目录
: "${IMAGE_ROOT:=${PROJECT_ROOT}/amazon18_data/Images}"
: "${MODEL_CACHE_DIR:=/scratch/yh4663/MQL4GRec/.cachemodels/clip}"
: "${CUDA_VISIBLE_DEVICES:=1}"                   # 需要时会被脚本使用

# 常用的目录派生（可按需添加）
SAVE_ROOT="${OUTPUT_PATH}"                       # 你脚本里用到的 save_root
