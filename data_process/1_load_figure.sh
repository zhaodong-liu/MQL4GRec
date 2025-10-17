#!/bin/bash
set -euo pipefail
DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/config.sh"

python "$DIR/load_all_figures.py" --dataset "$DATASET"
