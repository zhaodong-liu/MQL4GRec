#!/bin/bash
# Process Amazon18 data into splits & atomic files
# Usage:
#   bash 2_process.sh                          # 用 config.sh 默认
#   bash 2_process.sh Games                     # 临时换数据集
#   bash 2_process.sh Games /path/in /path/out  # 临时覆盖输入/输出根

set -euo pipefail

# 脚本所在目录（即 data_process/）
DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/config.sh"

# 1) 读取或覆盖 DATASET
DATASET="${1:-${DATASET:-Instruments}}"

# 2) 读取或覆盖输入/输出目录
INPUT_PATH_DEFAULT="${AMAZON18_INPUT_PATH:-amazon18_data}"
OUTPUT_PATH_DEFAULT="${OUTPUT_PATH:-MQL4GRec}"
INPUT_PATH="${2:-$INPUT_PATH_DEFAULT}"
OUTPUT_PATH="${3:-$OUTPUT_PATH_DEFAULT}"

# 3) 相对路径→相对脚本目录，确保目录一致
[[ "$INPUT_PATH"  != /* ]] && INPUT_PATH="$DIR/${INPUT_PATH#./}"
[[ "$OUTPUT_PATH" != /* ]] && OUTPUT_PATH="$DIR/${OUTPUT_PATH#./}"

mkdir -p "$OUTPUT_PATH"

echo "==> DATASET     : $DATASET"
echo "==> INPUT_PATH  : $INPUT_PATH"
echo "==> OUTPUT_PATH : $OUTPUT_PATH"
echo

python "$DIR/amazon18_data_process.py" \
  --dataset "$DATASET" \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH"
