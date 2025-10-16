# MQL4GRec

This is the code for the ICLR 2025 paper:  
[**Multimodal Quantitative Language for Generative Recommendation**](https://openreview.net/pdf?id=v7YrIjpkTF)

![alt text](figures/framework.png)

## Setup

> pytorch==2.1.0  
transformers <= 4.45.0  

- We found that different versions of **transformers** significantly impact convergence speed and performance under default parameters.  
For newer versions, parameter adjustments are required.

We tested different transformers versions with accelerate==0.28.0:  
- v4.47.0, 4.48.0, 4.50.0
![alt text](figures/2.png)  

- v4.38.2, 4.39.0, 4.40.0, 4.45.0
![alt text](figures/3.png)  

- v4.46.0 shows high training loss  
![alt text](figures/1.png)  

v4.43.0, 4.44.0 are incompatible with accelerate==0.28.0 - version change required.

## Quick Start

### Option 1: Automated Data Processing (Recommended)

**NEW**: We now provide automated scripts to download and process Amazon datasets from scratch!

```bash
cd data_process

# Run the full pipeline with one command (downloads data, images, generates embeddings)
bash run_full_pipeline.sh Instruments ./amazon18_data ./MQL4GRec_data 0

# Or download only
python 0_download_amazon_data.py --dataset Instruments --output_path ./amazon18_data
```

See `data_process/README.md` for detailed documentation.

**Available datasets:** Beauty, Fashion, Arts, Automotive, Books, Electronics, Food, Home, Instruments, Movies, Office, Pet, Sports, Toys, Games, and more.

### Option 2: Use Preprocessed Data

Preprocessed data, pretrained checkpoints, and training logs:
[Google Drive Folder](https://drive.google.com/drive/folders/1eewycbcAJ95atmF_V3bNchPIFDSw_TQC)

### Manual Data Processing Steps
```bash
cd data_process
```
1. Download Amazon dataset: `python 0_download_amazon_data.py --dataset Instruments`
2. Download images: `python load_all_figures.py --dataset Instruments`
3. Process data: `python amazon18_data_process.py --dataset Instruments`
4. Generate text embeddings: `python amazon_text_emb.py --dataset Instruments`
5. Generate image embeddings: `python clip_feature.py --dataset Instruments`

### Training the Quantitative Translator
```
cd index
bash script/run.sh          # Run training  
bash script/gen_code_dis.sh # Generate code  
```

### Pre-training
```
bash script/pretrain.sh
```

### Fine-tuning
```
bash finetune.sh
```

## Notes  
- Adjust file paths according to your local directory structure  

## Contributing  
PRs and issues are welcome!  

## License  
N/A  