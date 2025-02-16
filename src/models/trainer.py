from transformers import Trainer, TrainingArguments
import wandb
import os
from typing import Optional
from torch.utils.data import Dataset

class LlamaTrainer:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

    def get_training_args(self):
        return TrainingArguments(
            output_dir=self.config["model_config"]["output_dir"],
            num_train_epochs=self.config["model_config"]["num_epochs"],
            per_device_train_batch_size=self.config["model_config"]["batch_size"],
            per_device_eval_batch_size=self.config["model_config"]["batch_size"],
            gradient_accumulation_steps=self.config["model_config"]["gradient_accumulation_steps"],
            learning_rate=self.config["model_config"]["learning_rate"],
            weight_decay=self.config["model_config"]["weight_decay"],
            warmup_steps=self.config["model_config"]["warmup_steps"],
            logging_dir=os.path.join(self.config["model_config"]["output_dir"], "logs"),
            logging_steps=100,
            eval_steps=500,
            save_steps=1000,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            report_to="wandb"
        )

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ):
        # Initialize wandb
        wandb.init(
            project=self.config["wandb_config"]["project"],
            entity=self.config["wandb_config"]["entity"]
        )

        training_args = self.get_training_args()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Train the model
        trainer.train()
        
        # Save the final model and tokenizer
        trainer.save_model(os.path.join(self.config["model_config"]["output_dir"], "final_model"))
        self.tokenizer.save_pretrained(
            os.path.join(self.config["model_config"]["output_dir"], "final_model")
        )
        
        wandb.finish()

    def generate_text(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(
            f"[INST] {prompt} [/INST]",
            return_tensors="pt",
            truncation=True,
            max_length=self.config["model_config"]["max_length"]
        ).to(self.model.device)
        
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
