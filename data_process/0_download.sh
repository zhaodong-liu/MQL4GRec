#!/bin/bash
# Quick download script for Amazon datasets
# Usage:
#   bash 0_download.sh                 # 用 config.sh 的默认 DATASET 与目录
#   bash 0_download.sh Games           # 只临时换数据集
#   bash 0_download.sh Games /some/abs/path   # 临时换数据集与输出根路径

set -euo pipefail

# 脚本所在目录（即 data_process/）
DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/config.sh"

# 1) 取 DATASET：优先位置参数1，其次 config.sh，再次默认 Instruments
DATASET="${1:-${DATASET:-Instruments}}"

# 2) 取 OUTPUT_PATH：优先位置参数2，其次 config.sh 的 AMAZON18_INPUT_PATH，再次默认脚本同级的 amazon18_data
OUTPUT_PATH_DEFAULT="${AMAZON18_INPUT_PATH:-amazon18_data}"
OUTPUT_PATH="${2:-$OUTPUT_PATH_DEFAULT}"

# 3) 把相对路径规范化为相对于脚本目录的路径，确保“就在原来的 data_process 下”
if [[ "$OUTPUT_PATH" != /* ]]; then
  OUTPUT_PATH="$DIR/${OUTPUT_PATH#./}"
fi
mkdir -p "$OUTPUT_PATH"

echo "==> DATASET       : $DATASET"
echo "==> OUTPUT_PATH   : $OUTPUT_PATH"
echo "==> Downloader    : $DIR/0_download_amazon_data.py"
echo

python "$DIR/0_download_amazon_data.py" \
  --dataset "$DATASET" \
  --output_path "$OUTPUT_PATH"

echo
echo "✓ Download completed successfully!"
echo
echo "Next steps:"
echo "  1. Download images:     bash 1_load_figure.sh"
echo "  2. Process data:        bash 2_process.sh"
echo "  3. Extract text emb:    bash 3_get_text_emb.sh"
echo "  4. Extract image emb:   bash 4_get_image_emb.sh"
echo
