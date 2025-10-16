#!/bin/bash
# Quick download script for Amazon datasets
# Usage: bash 0_download.sh [dataset_name] [output_path]

DATASET=${1:-"Instruments"}
OUTPUT_PATH=${2:-"data_process/amazon18_data"}


python data_process/0_download_amazon_data.py \
    --dataset "$DATASET" \
    --output_path "$OUTPUT_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Download completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Download images:     bash 1_load_figure.sh"
    echo "  2. Process data:        bash 2_process.sh"
    echo "  3. Extract text emb:    bash 3_get_text_emb.sh"
    echo "  4. Extract image emb:   bash 4_get_image_emb.sh"
    echo ""
else
    echo ""
    echo "✗ Download failed. Please check errors above."
    echo ""
    exit 1
fi
