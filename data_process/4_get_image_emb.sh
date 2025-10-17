#!/bin/bash
set -euo pipefail
DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/config.sh"

export CUDA_VISIBLE_DEVICES   # 来自 config.sh

python "$DIR/clip_feature.py" \
  --image_root "$IMAGE_ROOT" \
  --save_root "$SAVE_ROOT" \
  --model_cache_dir "$MODEL_CACHE_DIR" \
  --dataset "$DATASET"
