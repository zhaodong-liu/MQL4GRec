# Usage Examples

This document provides practical examples for using the automated data processing pipeline.

## Example 1: Process Single Dataset (Musical Instruments)

The simplest way to get started:

```bash
cd data_process

# Run full pipeline (downloads everything, processes data, generates embeddings)
bash run_full_pipeline.sh

# This uses default parameters:
# - Dataset: Instruments
# - Raw data: ./amazon18_data
# - Processed data: ./MQL4GRec_data
# - GPU: 0
```

**Expected output:**
```
MQL4GRec_data/Instruments/
├── Instruments.inter.json
├── Instruments.train.inter
├── Instruments.valid.inter
├── Instruments.test.inter
├── Instruments.item.json
├── Instruments.user2id
├── Instruments.item2id
├── Instruments.emb-llama-td.npy      # Text embeddings
└── Instruments.emb-ViT-L-14.npy      # Image embeddings
```

**Time estimate:** ~2-4 hours (depends on network speed and GPU)

## Example 2: Process Custom Dataset

Process a different Amazon category:

```bash
# Process Arts & Crafts dataset on GPU 1
bash run_full_pipeline.sh Arts ./data ./processed 1
```

## Example 3: Download Only (No Processing)

Just download the raw Amazon data without processing:

```bash
# Download Musical Instruments
python 0_download_amazon_data.py --dataset Instruments --output_path ./raw_data

# Download multiple datasets at once
python 0_download_amazon_data.py \
    --dataset Instruments \
    --dataset Arts \
    --dataset Games \
    --output_path ./raw_data
```

## Example 4: Process Existing Data

If you already have raw Amazon data, just run processing steps:

```bash
# Assuming you have data in /datasets/amazon18/

# Step 1: Download images
python load_all_figures.py \
    --dataset Instruments \
    --meta_data_path /datasets/amazon18/Metadata \
    --review_data_path /datasets/amazon18/Review \
    --save_path /datasets/amazon18/Images

# Step 2: Process interactions
python amazon18_data_process.py \
    --dataset Instruments \
    --input_path /datasets/amazon18 \
    --output_path ./MQL4GRec_data

# Step 3: Generate text embeddings
export CUDA_VISIBLE_DEVICES=0
python amazon_text_emb.py \
    --dataset Instruments \
    --root ./MQL4GRec_data \
    --gpu_id 0

# Step 4: Generate image embeddings
python clip_feature.py \
    --dataset Instruments \
    --image_root /datasets/amazon18/Images \
    --save_root ./MQL4GRec_data \
    --gpu_id 0
```

## Example 5: Custom K-Core Filtering

Use stricter filtering (10-core instead of default 5-core):

```bash
# Download data first
python 0_download_amazon_data.py --dataset Books --output_path ./data

# Download images
python load_all_figures.py --dataset Books

# Process with 10-core filtering
python amazon18_data_process.py \
    --dataset Books \
    --user_k 10 \
    --item_k 10 \
    --input_path ./data \
    --output_path ./MQL4GRec_data

# Continue with embeddings...
```

## Example 6: Using Custom Models

### Use Different Text Model

```bash
# Use BERT instead of LLaMA
python amazon_text_emb.py \
    --dataset Instruments \
    --root ./MQL4GRec_data \
    --plm_name bert \
    --model_name_or_path bert-base-uncased \
    --max_sent_len 512
```

### Use Different Vision Model

```bash
# Use smaller CLIP model (ViT-B/32)
python clip_feature.py \
    --dataset Instruments \
    --backbone ViT-B/32 \
    --image_root ./data/Images \
    --save_root ./MQL4GRec_data
```

## Example 7: Process Multiple Datasets in Parallel

Process several datasets efficiently:

```bash
#!/bin/bash
# process_all.sh

DATASETS=("Instruments" "Arts" "Games")

for dataset in "${DATASETS[@]}"; do
    echo "Processing $dataset..."

    # Run in background
    bash run_full_pipeline.sh "$dataset" ./raw ./processed 0 &> "logs/${dataset}.log" &

    # Limit parallel jobs
    if [ $(jobs -r | wc -l) -ge 2 ]; then
        wait -n
    fi
done

wait
echo "All datasets processed!"
```

## Example 8: CPU-Only Processing

If you don't have a GPU:

```bash
# Download and process data (CPU-based)
python 0_download_amazon_data.py --dataset Beauty
python load_all_figures.py --dataset Beauty

# Process interactions (no GPU needed)
python amazon18_data_process.py --dataset Beauty

# Text embeddings on CPU (slow!)
python amazon_text_emb.py \
    --dataset Beauty \
    --gpu_id -1 \
    --max_sent_len 512

# Image embeddings on CPU
python clip_feature.py \
    --dataset Beauty \
    --gpu_id -1
```

## Example 9: Verify Downloaded Data

Check what datasets you've already downloaded:

```bash
# List available datasets
python 0_download_amazon_data.py --list-datasets

# Check downloaded files
ls -lh amazon18_data/Metadata/
ls -lh amazon18_data/Ratings/
ls -lh amazon18_data/Review/
```

## Example 10: Resume After Failure

If pipeline fails partway through:

```bash
# The scripts are idempotent - they skip already completed steps
# Just re-run the full pipeline
bash run_full_pipeline.sh Instruments

# Or run individual steps:

# Skip download if files exist
python 0_download_amazon_data.py --dataset Instruments  # Will skip existing files

# Resume image download (skips downloaded images)
python load_all_figures.py --dataset Instruments  # Continues from where it stopped

# Other steps will overwrite, so check before re-running
```

## Example 11: Small Dataset for Testing

Test the pipeline on a small dataset first:

```bash
# Gift Cards is one of the smallest datasets
bash run_full_pipeline.sh Gift ./test_data ./test_processed 0

# Typical size: ~10K items, ~50K users, ~200K interactions
# Processing time: ~30 minutes
```

## Example 12: Complete Workflow from Scratch

Full workflow from raw data to trained model:

```bash
# 1. Process data
cd data_process
bash run_full_pipeline.sh Instruments ./raw ./data 0

# 2. Train quantizers
cd ../index
bash scripts/run.sh

# 3. Generate codes
bash scripts/gen_code_dis.sh

# 4. Pre-train T5
cd ..
bash scripts/pretrain.sh

# 5. Fine-tune
bash scripts/finetune.sh

# 6. Test
python test.py --ckpt_path ./checkpoints/finetune --dataset Instruments
```

## Troubleshooting Examples

### Out of Memory

```bash
# Reduce batch size and sequence length
python amazon_text_emb.py \
    --dataset Instruments \
    --max_sent_len 512 \
    --gpu_id 0

# Or use smaller models
python clip_feature.py --backbone ViT-B/32
```

### Slow Image Downloads

```bash
# Download images in the background
nohup python load_all_figures.py --dataset Instruments > image_download.log 2>&1 &

# Check progress
tail -f image_download.log
```

### Check Data Statistics

```bash
# After processing, check dataset size
python -c "
import json
with open('./MQL4GRec_data/Instruments/Instruments.inter.json') as f:
    data = json.load(f)
    print(f'Users: {len(data)}')
    print(f'Avg interactions: {sum(len(v) for v in data.values()) / len(data):.1f}')
"
```

## Performance Tips

1. **Use SSD for image storage** - Image download creates many small files
2. **Parallel dataset processing** - Process different datasets on different GPUs
3. **Cache models** - Set `TRANSFORMERS_CACHE` and `TORCH_HOME` to fast storage
4. **Monitor GPU memory** - Use `nvidia-smi` to check utilization

```bash
# Set cache directories
export TRANSFORMERS_CACHE=/fast/cache/huggingface
export TORCH_HOME=/fast/cache/torch

# Run pipeline
bash run_full_pipeline.sh Instruments
```

## Next Steps

After data processing, see the main README for:
- Training quantizers (`index/README.md`)
- Pre-training T5 models
- Fine-tuning and evaluation

For questions, check `data_process/README.md` or open an issue.
