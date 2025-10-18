Dataset=Instruments,Arts,Games,CDs,Sports,Beauty

OUTPUT_DIR=./data/$Dataset

python -u index/generate_indices_distance.py \
  --dataset $Dataset \
  --device cuda:0 \
  --ckpt_path index/og/$Dataset/llama_256/best_collision_model.pth \
  --output_dir $OUTPUT_DIR \
  --output_file ${Dataset}.index_lemb.json


python -u index/generate_indices_distance.py \
    --dataset $Dataset \
    --device cuda:0 \
    --ckpt_path index/log/$Dataset/ViT-L-14_256/best_collision_model.pth \
    --output_dir $OUTPUT_DIR \
    --output_file ${Dataset}.index_vitemb.json \
    --content image

