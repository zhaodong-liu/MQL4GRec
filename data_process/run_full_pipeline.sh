#!/bin/bash
# Full automated pipeline for MQL4GRec data preprocessing
# This script runs all steps from downloading raw data to generating embeddings

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Default parameters (can be overridden via command line)
DATASET=${1:-"Instruments"}
RAW_DATA_ROOT=${2:-"./amazon18_data"}
PROCESSED_DATA_ROOT=${3:-"./MQL4GRec_data"}
GPU_ID=${4:-0}

# Model paths (update these to your local paths)
LLAMA_MODEL_PATH=${LLAMA_MODEL_PATH:-"huggyllama/llama-7b"}
LLAMA_CACHE_DIR=${LLAMA_CACHE_DIR:-"~/.cache/huggingface"}
CLIP_MODEL_NAME=${CLIP_MODEL_NAME:-"ViT-L/14"}
CLIP_CACHE_DIR=${CLIP_CACHE_DIR:-"~/.cache/clip"}

# ============================================================================
# Helper functions
# ============================================================================

print_header() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
    echo ""
}

print_step() {
    echo ""
    echo ">>> STEP $1: $2"
    echo ""
}

check_python_package() {
    python -c "import $1" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Error: Python package '$1' not found. Please install it first."
        echo "  pip install $1"
        exit 1
    fi
}

# ============================================================================
# Main Pipeline
# ============================================================================

print_header "MQL4GRec Full Data Processing Pipeline"

echo "Configuration:"
echo "  Dataset:              $DATASET"
echo "  Raw Data Root:        $RAW_DATA_ROOT"
echo "  Processed Data Root:  $PROCESSED_DATA_ROOT"
echo "  GPU ID:               $GPU_ID"
echo "  LLaMA Model:          $LLAMA_MODEL_PATH"
echo "  CLIP Model:           $CLIP_MODEL_NAME"
echo ""

read -p "Continue with these settings? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Check required packages
print_step "0" "Checking dependencies"
check_python_package "torch"
check_python_package "transformers"
check_python_package "PIL"
check_python_package "tqdm"
echo "✓ All required packages found"

# Step 1: Download Amazon dataset
print_step "1" "Downloading Amazon $DATASET dataset"
python 0_download_amazon_data.py \
    --dataset "$DATASET" \
    --output_path "$RAW_DATA_ROOT"

if [ $? -ne 0 ]; then
    echo "✗ Download failed"
    exit 1
fi
echo "✓ Download completed"

# Step 2: Download product images
print_step "2" "Downloading product images"
python load_all_figures.py \
    --dataset "$DATASET" \
    --meta_data_path "$RAW_DATA_ROOT/Metadata" \
    --rating_data_path "$RAW_DATA_ROOT/Ratings" \
    --review_data_path "$RAW_DATA_ROOT/Review" \
    --save_path "$RAW_DATA_ROOT/Images"

if [ $? -ne 0 ]; then
    echo "✗ Image download failed"
    exit 1
fi
echo "✓ Image download completed"

# Step 3: Process interaction data and metadata
print_step "3" "Processing interaction data and metadata"
python amazon18_data_process.py \
    --dataset "$DATASET" \
    --input_path "$RAW_DATA_ROOT" \
    --output_path "$PROCESSED_DATA_ROOT"

if [ $? -ne 0 ]; then
    echo "✗ Data processing failed"
    exit 1
fi
echo "✓ Data processing completed"

# Step 4: Generate text embeddings with LLaMA
print_step "4" "Generating text embeddings with LLaMA"
export CUDA_VISIBLE_DEVICES=$GPU_ID
python amazon_text_emb.py \
    --dataset "$DATASET" \
    --root "$PROCESSED_DATA_ROOT" \
    --gpu_id 0 \
    --plm_name "llama" \
    --model_name_or_path "$LLAMA_MODEL_PATH" \
    --model_cache_dir "$LLAMA_CACHE_DIR" \
    --max_sent_len 2048

if [ $? -ne 0 ]; then
    echo "✗ Text embedding generation failed"
    exit 1
fi
echo "✓ Text embeddings generated"

# Step 5: Generate image embeddings with CLIP
print_step "5" "Generating image embeddings with CLIP"
export CUDA_VISIBLE_DEVICES=$GPU_ID
python clip_feature.py \
    --dataset "$DATASET" \
    --image_root "$RAW_DATA_ROOT/Images" \
    --save_root "$PROCESSED_DATA_ROOT" \
    --gpu_id 0 \
    --backbone "$CLIP_MODEL_NAME" \
    --model_cache_dir "$CLIP_CACHE_DIR"

if [ $? -ne 0 ]; then
    echo "✗ Image embedding generation failed"
    exit 1
fi
echo "✓ Image embeddings generated"

# ============================================================================
# Summary
# ============================================================================

print_header "Pipeline Completed Successfully!"

DATASET_DIR="$PROCESSED_DATA_ROOT/$DATASET"

echo "Generated files in $DATASET_DIR:"
echo ""
echo "  Interaction data:"
echo "    - $DATASET.inter.json"
echo "    - $DATASET.train.inter"
echo "    - $DATASET.valid.inter"
echo "    - $DATASET.test.inter"
echo ""
echo "  Metadata:"
echo "    - $DATASET.item.json"
echo "    - $DATASET.user2id"
echo "    - $DATASET.item2id"
echo ""
echo "  Embeddings:"
echo "    - $DATASET.emb-llama-td.npy      (text embeddings)"
echo "    - $DATASET.emb-ViT-L-14.npy      (image embeddings)"
echo ""
echo "Next steps:"
echo "  1. Train RQVAE quantizers:  cd ../index && bash scripts/run.sh"
echo "  2. Generate quantized codes: cd ../index && bash scripts/gen_code_dis.sh"
echo "  3. Pre-train T5 model:       bash scripts/pretrain.sh"
echo "  4. Fine-tune on target data: bash scripts/finetune.sh"
echo ""
print_header "All Done!"
