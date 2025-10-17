#!/bin/bash
set -euo pipefail
# 让脚本无论从哪里执行都能找到 config.sh
DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/config.sh"
# 允许用位置参数临时覆盖 DATASET（可选）
DATASET="${1:-$DATASET}"
# 从 config.sh 继承 GPU 设置（也可在外部临时覆盖）
export CUDA_VISIBLE_DEVICES
python "$DIR/amazon_text_emb.py" --dataset "$DATASET"
