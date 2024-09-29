
# QL4GRec

This is the code for the ICLR 2025 submission: Multimodal Quantitative Language for Generative Recommendation.

## Setup

Before running the scripts, ensure you have installed all the necessary dependencies and requirements.

## Quick Start

Follow these steps to get started:

### Training the Quantitative Translator

1. Navigate to the `index` folder:
   ```
   cd index
   ```
2. Run the training script:
   ```
   bash script/run.sh
   ```
3. Generate code distributions:
   ```
   bash script/gen_code_dis.sh
   ```

### Pre-training

1. Start the pre-training process:
   ```
   bash script/pretrain.sh
   ```

### Fine-tuning

1. Perform fine-tuning:
   ```
   bash finetune.sh
   ```

## Notes

- Ensure that you have the necessary permissions to execute shell scripts.
- Modify the script paths if necessary to match your local directory structure.
- Check that all scripts are executable by running:
   ```
   chmod +x script_name.sh
   ```

## Contributing

If you have any suggestions or fixes, please create a pull request or open an issue.

## License

N/A
