import yaml
import os
from data.dataset import LlamaDataset
from data.preprocessor import DataPreprocessor
from models.model import LlamaModel
from models.trainer import LlamaTrainer
from utils.logger import setup_logger

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Setup logger
    logger = setup_logger("llama_finetuning")
    logger.info("Starting Llama fine-tuning process")
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(config)
    
    try:
        # Load and preprocess data
        logger.info("Loading data from CSV")
        raw_df = preprocessor.load_data(config["data_config"]["raw_data_path"])
        processed_data = preprocessor.preprocess(raw_df)
        
        # Split data
        logger.info("Splitting data into train and validation sets")
        train_data, val_data = preprocessor.split_data(processed_data)
        
        # Save processed data
        logger.info("Saving processed data")
        preprocessor.save_data(train_data, config["data_config"]["train_path"])
        preprocessor.save_data(val_data, config["data_config"]["val_path"])
        
    # Initialize model
        llama_model = LlamaModel(config)
        tokenizer, model = llama_model.load_model()
        
        # Create datasets
        train_dataset = LlamaDataset(
            train_data,
            tokenizer,
            config["model_config"]["max_length"]
        )
        val_dataset = LlamaDataset(
            val_data,
            tokenizer,
            config["model_config"]["max_length"]
        )
        
        # Initialize trainer
        trainer = LlamaTrainer(config, model, tokenizer)
        
        # Train model
        logger.info("Starting training")
        trainer.train(train_dataset, val_dataset)
        logger.info("Training completed")
        
        # Test generation
        test_prompt = "What is machine learning?"
        generated_text = trainer.generate_text(test_prompt)
        logger.info(f"Generated text for prompt '{test_prompt}':\n{generated_text}")
      
    except Exception as e:
        logger.error(f"Error during data processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()