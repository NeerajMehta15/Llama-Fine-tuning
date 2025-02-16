#!/usr/bin/env python3
import os
import yaml
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the base directory structure for the project"""
    # Define the directory structure
    directories = [
        "config",
        "data/raw",
        "data/processed",
        "logs",
        "models/checkpoints",
        "notebooks",
        "src/data",
        "src/models",
        "src/utils",
    ]

    # Create each directory
    for directory in directories:
        Path(f"llama_finetuning/{directory}").mkdir(parents=True, exist_ok=True)

def create_init_files():
    """Create __init__.py files in all Python package directories"""
    python_dirs = [
        "src",
        "src/data",
        "src/models",
        "src/utils"
    ]
    
    for directory in python_dirs:
        init_file = Path(f"llama_finetuning/{directory}/__init__.py")
        init_file.touch()

def create_config_yaml():
    """Create the config.yaml file with default configurations"""
    config = {
        "model_config": {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "max_length": 512,
            "batch_size": 4,
            "learning_rate": 2e-5,
            "num_epochs": 3,
            "gradient_accumulation_steps": 8,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "output_dir": "models/checkpoints"
        },
        "data_config": {
            "train_path": "data/processed/train.json",
            "val_path": "data/processed/val.json",
            "raw_data_path": "data/raw/data.json"
        },
        "wandb_config": {
            "project": "llama-finetuning",
            "entity": "your-username"
        }
    }
    
    with open("llama_finetuning/config/config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def create_requirements():
    """Create requirements.txt with necessary dependencies"""
    requirements = """torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
pyyaml>=6.0
pandas>=1.5.0
numpy>=1.24.0
tqdm>=4.65.0
accelerate>=0.20.0
bitsandbytes>=0.41.0"""
    
    with open("llama_finetuning/requirements.txt", 'w') as f:
        f.write(requirements)

def create_readme():
    """Create README.md with project documentation"""
    readme = """# Llama Fine-tuning Project

This project provides a modular implementation for fine-tuning Llama-2-7b models.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prerequisites:
- Access to Llama 2 through Hugging Face
- Hugging Face login: `huggingface-cli login`
- Weights & Biases account: `wandb login`

3. Prepare your data:
- Place your training data in `data/raw/data.json`

4. Run training:
```bash
python src/main.py
```

## Project Structure

```
llama_finetuning/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── logs/
├── models/
│   └── checkpoints/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   └── utils/
└── requirements.txt
```
"""
    
    with open("llama_finetuning/README.md", 'w') as f:
        f.write(readme)

def create_gitignore():
    """Create .gitignore file"""
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environments
.env
.venv
env/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
logs/
models/checkpoints/
data/processed/
wandb/
"""
    
    with open("llama_finetuning/.gitignore", 'w') as f:
        f.write(gitignore)

def main():
    # Create project root directory
    project_root = Path("llama_finetuning")
    if project_root.exists():
        response = input("Directory 'llama_finetuning' already exists. Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
        shutil.rmtree(project_root)
    
    print("Creating project structure...")
    
    # Create directories and files
    create_directory_structure()
    create_init_files()
    create_config_yaml()
    create_requirements()
    create_readme()
    create_gitignore()
    
    print("""Project structure created successfully!""")

if __name__ == "__main__":
    main()