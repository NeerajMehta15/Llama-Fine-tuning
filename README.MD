# Llama 2 Fine-tuning Project

A modular implementation for fine-tuning Llama-2-7b models with custom datasets. This project provides a complete pipeline for data preprocessing, model training, and evaluation.

## Features

- Complete fine-tuning pipeline for Llama 2 models
- CSV data support with flexible column mapping
- Modular architecture for easy customization
- Comprehensive logging and error handling
- Weights & Biases integration for experiment tracking
- Automatic train/validation split
- GPU memory optimization

## Data Format

Prepare your data as a CSV file with the following format:
Place this file in `data/raw/data.csv`.

## Configuration

Edit `config/config.yaml` to customize your training:

```yaml
model_config:
  model_name: "meta-llama/Llama-2-7b-hf"
  max_length: 512
  batch_size: 4
  learning_rate: 2e-5
  num_epochs: 3
  gradient_accumulation_steps: 8
  warmup_steps: 100
  weight_decay: 0.01
  output_dir: "models/checkpoints"
  
data_config:
  train_path: "data/processed/train.csv"
  val_path: "data/processed/val.csv"
  raw_data_path: "data/raw/data.csv"
  prompt_column: "prompt"
  completion_column: "completion"
  
wandb_config:
  project: "llama-finetuning"
  entity: "your-username"
```

## Usage

1. Prepare your data in CSV format and place it in `data/raw/data.csv`

2. Adjust configuration in `config/config.yaml`

3. Run the training:
```bash
python src/main.py
```

4. Monitor training:
- Check console output for progress
- View logs in `logs/` directory
- Monitor on Weights & Biases dashboard (if configured)
