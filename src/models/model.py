from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LlamaModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_config"]["model_name"]
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_config"]["model_name"],
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        return self.tokenizer, self.model