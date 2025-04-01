# MQL4GRec

This is the code for the ICLR 2025 paper:  
[**Multimodal Quantitative Language for Generative Recommendation**](https://openreview.net/pdf?id=v7YrIjpkTF)

## Setup

Before running the scripts, ensure you have installed all the necessary dependencies and requirements.

## Quick Start

### Data Processing
```
cd data_process
```
1. Download images  
2. Process data so that each item corresponds to one image and one text description  
3. Obtain text embeddings
4. Obtain image embeddings    

Preprocessed data, pretrained checkpoints, and training logs:  
[Google Drive Folder](https://drive.google.com/drive/folders/1eewycbcAJ95atmF_V3bNchPIFDSw_TQC)

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
- Ensure execution permissions for scripts (`chmod +x *.sh`)
- Adjust paths to match your local directory structure

## Contributing
PRs and issues are welcome!

## License
N/A