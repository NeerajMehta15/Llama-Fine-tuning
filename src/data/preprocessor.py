import pandas as pd
from typing import List, Dict
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.prompt_col = config["data_config"]["prompt_column"]
        self.completion_col = config["data_config"]["completion_column"]

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        return pd.read_csv(file_path)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data"""
        # Clean whitespace
        data[self.prompt_col] = data[self.prompt_col].str.strip()
        data[self.completion_col] = data[self.completion_col].str.strip()
        
        # Remove rows with missing values
        data = data.dropna(subset=[self.prompt_col, self.completion_col])
        
        # Convert to list of dictionaries for easier handling
        processed_data = data.to_dict('records')
        
        return processed_data

    def save_data(self, data: List[Dict], output_path: str):
        """Save data to CSV file"""
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

    def split_data(self, data: List[Dict], test_size: float = 0.1, random_state: int = 42):
        """Split data into train and validation sets"""
        train_data, val_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state
        )
        return train_data, val_data