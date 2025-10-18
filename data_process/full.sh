<<<<<<< HEAD
#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --account=pr_119_tandon_priority
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yh46639@nyu.edu
#SBATCH --requeue

=======
>>>>>>> origin/master


DATASETS=(CDs Sports Games Beauty Arts Instruments)


export CUDA_VISIBLE_DEVICES

for DATASET in "${DATASETS[@]}"; do
  python data_process/0_download_amazon_data.py \
    --dataset "$DATASET" \
    --output_path "data_process/amazon18_data"

  python data_process/load_all_figures.py --dataset "$DATASET"

  python data_process/amazon18_data_process.py \
    --dataset "$DATASET" \
    --input_path data_process/amazon18_data \
    --output_path data_process/MQL4GRec 


  python data_process/amazon_text_emb.py --dataset "$DATASET"

  python data_process/clip_feature.py \
    --image_root data_process/amazon18_data/Images \
    --save_root data_process/MQL4GRec \
    --model_cache_dir cache_models/clip \
    --dataset "$DATASET"
done
