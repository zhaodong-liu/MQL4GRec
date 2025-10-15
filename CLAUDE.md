# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MQL4GRec is a multimodal recommendation system using T5-based generative models with quantized indices for items. The system converts item embeddings (text and image) into discrete codes and uses these for sequential recommendation.

**Paper**: "Multimodal Quantitative Language for Generative Recommendation" (ICLR 2025)

## Key Architecture Components

### Three-Stage Pipeline

1. **Quantitative Translator Training** (`index/` directory)
   - Trains RQVAE (Residual Quantized Variational AutoEncoder) models to convert continuous embeddings into discrete codes
   - Two separate models: one for text embeddings (LLaMA), one for image embeddings (ViT-L-14)
   - Uses Sinkhorn-Knopp algorithm for quantization with multiple codebook levels
   - Outputs: quantized indices mapping item IDs to discrete token sequences

2. **Pre-training** (`pretrain.py`)
   - Trains T5 model on multiple datasets simultaneously
   - Uses soft prompts for different tasks (seqrec, seqimage)
   - Adds custom tokens for quantized codes to the tokenizer vocabulary
   - Format: `<a_{code}>`, `<b_{code}>`, `<c_{code}>`, `<d_{code}>` for text; `<A_{code}>`, `<B_{code}>`, `<C_{code}>`, `<D_{code}>` for images

3. **Fine-tuning** (`finetune.py`)
   - Fine-tunes on target dataset with multiple tasks (seqrec, seqimage, item2image, image2item, fusionseqrec)
   - Supports early stopping and best model selection
   - Uses distributed training with DDP

### Data Format

- **`.inter.json`**: User-item interaction sequences (dict of user_id -> [item_ids])
- **`.index_lemb.json`**: Item ID to text embedding codes (list of tokens like `["<a_1>", "<b_2>", ...]`)
- **`.index_vitemb.json`**: Item ID to image embedding codes
- **`.item.json`**: Item metadata

### Task Types

The system supports multiple training tasks controlled by `--tasks` parameter:
- `seqrec`: Sequential recommendation with item codes
- `seqimage`: Sequential recommendation with image codes
- `item2image` / `image2item`: Translation between item and image codes
- `fusionseqrec`: Fusion-based sequential recommendation with both modalities
- `fgfusionseqrec`: Fine-grained fusion with separate prompts for each modality

## Common Development Commands

### Training Quantitative Translator

```bash
cd index
bash scripts/run.sh          # Train RQVAE models for text and image embeddings
bash scripts/gen_code_dis.sh # Generate quantized codes for items
```

Key parameters in `index/main.py` and `index/main_mul.py`:
- `--num_emb_list`: Codebook sizes for each quantization level (e.g., `256 256 256 256`)
- `--sk_epsilons`: Sinkhorn-Knopp epsilon values for each level
- `--data_root`: Root directory containing `.emb-*.npy` embedding files
- `--embedding_file`: Embedding file suffix (e.g., `.emb-llama-td.npy`, `.emb-ViT-L-14.npy`)

### Pre-training

```bash
bash scripts/pretrain.sh
```

Uses `torchrun` for distributed training (default: 4 GPUs). Key parameters:
- `--pretrain_datasets`: Comma-separated dataset names for multi-dataset training
- `--tasks`: Training tasks (e.g., `seqrec,seqimage`)
- `--train_data_mode`: Data generation mode (0=all sequences, 1=max-length only)

### Fine-tuning

```bash
bash scripts/finetune.sh
```

Important: Set `load_model_name` to the pretrained checkpoint path. Script includes:
1. Fine-tuning with DDP (default: 2 GPUs)
2. Testing with `test_ddp_save.py`
3. Ensemble inference with `ensemble.py`

### Testing

Single GPU:
```bash
python test.py \
  --ckpt_path <checkpoint_dir> \
  --dataset <dataset_name> \
  --test_task seqrec \
  --num_beams 20 \
  --filter_items
```

Distributed (DDP):
```bash
torchrun --nproc_per_node=2 test_ddp_save.py \
  --ckpt_path <checkpoint_dir> \
  --num_beams 20 \
  --filter_items
```

## Important Implementation Details

### Transformers Version Compatibility

**Critical**: The codebase is highly sensitive to transformers version. Use `transformers<=4.45.0` with `pytorch==2.1.0`.
- Versions 4.46.0+ show high training loss
- Versions 4.43.0-4.44.0 incompatible with `accelerate==0.28.0`
- Best tested: 4.38.2, 4.39.0, 4.40.0, 4.45.0

### Token Management

The system dynamically adds quantized code tokens to the T5 tokenizer:
- `BaseDataset.get_all_tokens()` generates the full token vocabulary
- `BaseDataset.get_new_tokens()` returns tokens from actual index files
- Tokenizer is resized with `model.resize_token_embeddings(len(tokenizer))`

### Constrained Generation

Testing uses Trie-based prefix constraints to ensure valid item code generation:
- `generation_trie.Trie`: Implements prefix tree for allowed token sequences
- `prefix_allowed_tokens_fn()`: HuggingFace callback for constrained beam search
- Set `--filter_items` to filter predictions to valid item IDs only

### Data Sampling Modes (`train_data_mode`)

- `0`: All subsequences (sliding window over history)
- `1`: Only max-length sequences
- `2`: Random end position with max-length history
- `3`: Random end, then max-length history
- `4+`: Random start and end positions

### Distributed Training

Both pretrain and finetune use PyTorch DDP via `torchrun`:
- Check `WORLD_SIZE` environment variable to detect DDP mode
- `LOCAL_RANK` determines device assignment
- Set `ddp_find_unused_parameters=False` for efficiency

## File Paths to Adjust

When running locally, update paths in shell scripts:
- `scripts/pretrain.sh`: `--data_path`, base model in `--base_model`
- `scripts/finetune.sh`: `load_model_name` to pretrained checkpoint
- `index/scripts/run.sh`: `--data_root` to embedding directory
- `index/scripts/gen_code_dis.sh`: `--ckpt_path` to trained RQVAE models

## Project Structure

```
MQL4GRec/
├── index/                    # Quantitative translator (RQVAE)
│   ├── models/              # VQ, RQ, RQVAE implementations
│   ├── generate_indices_distance.py  # Code generation
│   └── main_mul.py          # Multi-dataset training
├── data_process/            # Dataset preprocessing utilities
├── data.py                  # Dataset classes (SeqRecDataset, FusionSeqRecDataset, etc.)
├── pretrain.py              # Multi-dataset pre-training
├── finetune.py              # Single-dataset fine-tuning
├── test.py / test_ddp.py    # Inference scripts
├── collator.py              # Data collation for training/testing
├── evaluate.py              # Metrics (Hit@K, NDCG@K)
├── ensemble.py              # Multi-task ensemble inference
└── utils.py                 # Argument parsing, data loading utilities
```
