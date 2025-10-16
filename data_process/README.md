# MQL4GRec Data Processing Pipeline

This directory contains scripts for preprocessing Amazon Review Dataset (2018) for the MQL4GRec recommendation system.

## Quick Start

### Option 1: Full Automated Pipeline (Recommended)

Run the complete pipeline with a single command:

```bash
# Download and process Musical Instruments dataset (default)
bash run_full_pipeline.sh

# Or specify custom parameters
bash run_full_pipeline.sh <dataset> <raw_data_dir> <processed_data_dir> <gpu_id>

# Example: Process Arts dataset on GPU 1
bash run_full_pipeline.sh Arts ./raw_data ./processed_data 1
```

### Option 2: Step-by-Step Execution

Run each step individually for more control:

```bash
# Step 0: Download Amazon dataset
python 0_download_amazon_data.py --dataset Instruments --output_path ./amazon18_data

# Step 1: Download product images
python load_all_figures.py --dataset Instruments

# Step 2: Process interactions and metadata
python amazon18_data_process.py --dataset Instruments

# Step 3: Generate text embeddings
python amazon_text_emb.py --dataset Instruments

# Step 4: Generate image embeddings
python clip_feature.py --dataset Instruments
```

## Available Datasets

The downloader supports all Amazon 2018 datasets:

| Short Name  | Full Category Name                  |
|-------------|-------------------------------------|
| Beauty      | All_Beauty                          |
| Fashion     | AMAZON_FASHION                      |
| Appliances  | Appliances                          |
| Arts        | Arts_Crafts_and_Sewing              |
| Automotive  | Automotive                          |
| Books       | Books                               |
| CDs         | CDs_and_Vinyl                       |
| Cell        | Cell_Phones_and_Accessories         |
| Clothing    | Clothing_Shoes_and_Jewelry          |
| Electronics | Electronics                         |
| Food        | Grocery_and_Gourmet_Food            |
| Home        | Home_and_Kitchen                    |
| Instruments | Musical_Instruments                 |
| Movies      | Movies_and_TV                       |
| Office      | Office_Products                     |
| Pet         | Pet_Supplies                        |
| Sports      | Sports_and_Outdoors                 |
| Toys        | Toys_and_Games                      |
| Games       | Video_Games                         |
| ... and more (see `utils.py` for complete list) |

To list all available datasets:
```bash
python 0_download_amazon_data.py --list-datasets
```

## Pipeline Stages Explained

### Stage 0: Download Raw Data

**Script:** `0_download_amazon_data.py`

Downloads three files from UCSD Amazon dataset repository:
- **Metadata** (`meta_*.json.gz`): Item information (title, description, brand, categories, image URLs)
- **Ratings** (`*.csv`): User-item interactions (user_id, item_id, rating, timestamp)
- **Reviews** (`*_5.json.gz`): 5-core reviews (users and items with ≥5 interactions)

**Output:**
```
amazon18_data/
├── Metadata/
│   └── meta_{category}.json.gz
├── Ratings/
│   └── {category}.csv
└── Review/
    └── {category}_5.json.gz
```

**Options:**
```bash
# Download single dataset
python 0_download_amazon_data.py --dataset Instruments --output_path ./data

# Download multiple datasets
python 0_download_amazon_data.py --dataset Instruments --dataset Arts --dataset Games

# Custom output directory
python 0_download_amazon_data.py --dataset Beauty --output_path /path/to/data
```

### Stage 1: Download Product Images

**Script:** `load_all_figures.py`

Downloads actual product images from URLs in metadata. Only downloads images for items that appear in the ratings data.

**Key features:**
- Downloads high-resolution images (`imageURLHighRes` field)
- Validates JPG files (checks for proper EOF marker)
- Skips already downloaded images
- Creates mapping file `{dataset}_images_info.json`

**Output:**
```
amazon18_data/Images/
├── {dataset}/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── {dataset}_images_info.json  # item_id -> image_filename mapping
```

**Usage:**
```bash
python load_all_figures.py \
    --dataset Instruments \
    --meta_data_path ./amazon18_data/Metadata \
    --review_data_path ./amazon18_data/Review \
    --save_path ./amazon18_data/Images
```

### Stage 2: Process Interactions & Metadata

**Script:** `amazon18_data_process.py`

Processes raw data into format required by MQL4GRec:
1. **Filter items:** Keep only items with images and metadata
2. **K-core filtering:** Default 5-core (users and items with ≥5 interactions)
3. **Chronological sorting:** Order interactions by timestamp
4. **Train/Valid/Test split:** Leave-one-out strategy
5. **Clean metadata:** Extract and clean title, description, brand, categories

**Output:**
```
MQL4GRec_data/{dataset}/
├── {dataset}.inter.json           # All interactions (user_id -> [item_ids])
├── {dataset}.train.inter          # Training interactions
├── {dataset}.valid.inter          # Validation interactions
├── {dataset}.test.inter           # Test interactions
├── {dataset}.item.json            # Item metadata (id -> {title, desc, ...})
├── {dataset}.user2id              # User remapping (original_id -> int_id)
└── {dataset}.item2id              # Item remapping (original_id -> int_id)
```

**K-core filtering:**
```bash
# Default 5-core
python amazon18_data_process.py --dataset Instruments --user_k 5 --item_k 5

# More strict 10-core
python amazon18_data_process.py --dataset Instruments --user_k 10 --item_k 10
```

**File formats:**

`.inter` files (tab-separated):
```
user_id:token    item_id_list:token_seq    item_id:token
0                1 5 12 8                  15
```

`.item.json`:
```json
{
  "0": {
    "title": "Guitar Strings Set",
    "description": "High quality steel strings...",
    "brand": "Ernie Ball",
    "categories": "Musical Instruments,Guitars,Strings"
  }
}
```

### Stage 3: Generate Text Embeddings

**Script:** `amazon_text_emb.py`

Encodes item text (title + description) using LLaMA-7B language model.

**Process:**
1. Load item metadata
2. Concatenate title and description
3. Tokenize with LLaMA tokenizer (max 2048 tokens)
4. Extract last hidden states
5. Mean pooling over sequence length
6. Save as NumPy array

**Output:**
```
MQL4GRec_data/{dataset}/
└── {dataset}.emb-llama-td.npy     # Shape: [num_items, 4096]
```

**Usage:**
```bash
export CUDA_VISIBLE_DEVICES=0

python amazon_text_emb.py \
    --dataset Instruments \
    --root ./MQL4GRec_data \
    --gpu_id 0 \
    --plm_name llama \
    --model_name_or_path huggyllama/llama-7b \
    --model_cache_dir ~/.cache/huggingface \
    --max_sent_len 2048
```

**Model options:**
- `huggyllama/llama-7b` (default, 4096-dim embeddings)
- `bert-base-uncased` (768-dim)
- Any HuggingFace model with AutoModel support

### Stage 4: Generate Image Embeddings

**Script:** `clip_feature.py`

Extracts visual features from product images using CLIP ViT-L/14.

**Process:**
1. Load raw JPG images
2. Apply CLIP preprocessing (resize, normalize)
3. Extract image features with ViT encoder
4. Save as NumPy array

**Output:**
```
MQL4GRec_data/{dataset}/
└── {dataset}.emb-ViT-L-14.npy     # Shape: [num_items, 768]
```

**Usage:**
```bash
export CUDA_VISIBLE_DEVICES=0

python clip_feature.py \
    --dataset Instruments \
    --image_root ./amazon18_data/Images \
    --save_root ./MQL4GRec_data \
    --gpu_id 0 \
    --backbone ViT-L/14 \
    --model_cache_dir ~/.cache/clip
```

**Backbone options:**
- `ViT-L/14` (default, 768-dim) - Recommended
- `ViT-B/32` (512-dim) - Faster but lower quality
- `RN50` (1024-dim) - ResNet-50 based

## After Data Processing

Once you have the embeddings, proceed to quantization training:

```bash
# Navigate to index directory
cd ../index

# Train RQVAE quantizers for both text and image embeddings
bash scripts/run.sh

# Generate quantized codes
bash scripts/gen_code_dis.sh
```

This will create:
- `{dataset}.index_lemb.json` - Text embedding codes
- `{dataset}.index_vitemb.json` - Image embedding codes

Then you can train the T5 model:

```bash
# Pre-training
cd ..
bash scripts/pretrain.sh

# Fine-tuning
bash scripts/finetune.sh
```

## Troubleshooting

### Download Issues

**Problem:** Downloads fail or timeout
```bash
# Retry individual files
python 0_download_amazon_data.py --dataset Instruments --output_path ./data
```

**Problem:** Corrupted gzip files
```bash
# The script automatically verifies and re-downloads corrupted files
# If issues persist, manually delete the corrupted file and retry
rm ./amazon18_data/Metadata/meta_*.json.gz
python 0_download_amazon_data.py --dataset Instruments
```

### Memory Issues

**Problem:** OOM when generating embeddings

For text embeddings:
```bash
# Reduce max sequence length
python amazon_text_emb.py --max_sent_len 512

# Or process in smaller batches (edit batch_size in script)
```

For image embeddings:
```bash
# Use smaller CLIP model
python clip_feature.py --backbone ViT-B/32
```

### Image Download Issues

**Problem:** Many images fail to download

This is normal - some Amazon image URLs are expired. The script will:
- Skip invalid images
- Report coverage rate at the end
- Only keep items with valid images in subsequent processing

Typical coverage rates: 70-95% depending on dataset

### GPU Out of Memory

```bash
# Use CPU instead (slower)
python amazon_text_emb.py --gpu_id -1

# Or use smaller models
python clip_feature.py --backbone ViT-B/32
```

## Requirements

```bash
pip install torch torchvision torchaudio
pip install transformers>=4.30.0
pip install pillow
pip install tqdm
pip install numpy
```

For CLIP:
```bash
pip install ftfy regex
# Or use the included clip module in data_process/clip/
```

## Data Statistics

Example statistics for Musical Instruments dataset:

| Metric | Before K-Core | After 5-Core |
|--------|---------------|--------------|
| Users | ~500K | ~50K |
| Items | ~900K | ~30K |
| Interactions | ~10M | ~1M |
| Avg. seq length | ~20 | ~25 |
| Items with images | ~85% | ~90% |

## Citation

If you use this data processing pipeline, please cite:

```bibtex
@inproceedings{mql4grec,
  title={Multimodal Quantitative Language for Generative Recommendation},
  booktitle={ICLR},
  year={2025}
}
```

## License

This pipeline processes data from the Amazon Review Dataset (2018):
- Dataset source: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
- Dataset paper: https://arxiv.org/abs/1809.07686

Please follow the original dataset's license terms.
