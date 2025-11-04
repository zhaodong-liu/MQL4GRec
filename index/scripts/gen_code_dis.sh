#!/bin/bash

# Dataset=Instruments,Arts,Games,CDs,Sports,Beauty
Dataset=Arts

# 数据根目录（embedding文件所在路径）
DATA_ROOT=data_process/MQL4GRec

# 输出目录
OUTPUT_DIR=./data/$Dataset
mkdir -p $OUTPUT_DIR

# 生成文本embedding的索引（LLaMA）
echo "Generating text embedding indices for $Dataset..."
python -u index/generate_indices_distance.py \
  --dataset $Dataset \
  --device cuda:0 \
  --ckpt_path index/log/$Dataset/llama_256/best_collision_model.pth \
  --data_root $DATA_ROOT \
  --embedding_file .emb-llama-td.npy \
  --output_dir $OUTPUT_DIR \
  --output_file ${Dataset}.index_lemb.json

echo "Text indices generated successfully!"

# 生成图像embedding的索引（ViT-L-14）
echo "Generating image embedding indices for $Dataset..."
python -u index/generate_indices_distance.py \
  --dataset $Dataset \
  --device cuda:0 \
  --ckpt_path index/log/$Dataset/ViT-L-14_256/best_collision_model.pth \
  --data_root $DATA_ROOT \
  --embedding_file .emb-ViT-L-14.npy \
  --output_dir $OUTPUT_DIR \
  --output_file ${Dataset}.index_vitemb.json \
  --content image

echo "Image indices generated successfully!"
echo "All indices saved to $OUTPUT_DIR"