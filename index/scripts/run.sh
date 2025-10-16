

Model=llama
Code_num=256

# Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'
Datasets='Instruments'

OUTPUT_DIR=log/$Datasets/${Model}_${Code_num}
mkdir -p $OUTPUT_DIR

python -u main_mul.py \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:0 \
  --data_root /scratch/zl4789/MQL4GRec/data \
  --embedding_file .emb-llama-td.npy \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --eval_step 2 \
  --batch_size 2048 \
  --epochs 500 > $OUTPUT_DIR/train.log

Model=ViT-L-14
Code_num=256

Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'

OUTPUT_DIR=log/$Datasets/${Model}_${Code_num}
mkdir -p $OUTPUT_DIR

python -u main_mul.py \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:0 \
  --data_root /scratch/zl4789/MQL4GRec/data \
  --embedding_file .emb-ViT-L-14.npy \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --eval_step 2 \
  --batch_size 2048 \
  --epochs 500 > $OUTPUT_DIR/train.log