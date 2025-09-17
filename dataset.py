import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import config
from utils import get_logger

logger = get_logger(__name__)

class SVDataset(Dataset):
    def __init__(self, tsv_path: str, tokenizer_name: str, max_length: int):
        logger.info(f"Loading data from {tsv_path}")
        self.data = pd.read_csv(tsv_path, sep="\t")
        logger.info(f"Initializing tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        ref_seq = row['ref_seq']
        alt_seq = row['alt_seq']
        label = int(row['label'])

        # DNABERT-2 expects sequences to be tokenized as pairs
        inputs = self.tokenizer(
            ref_seq,
            alt_seq,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt" # Return PyTorch tensors
        )

        # Squeeze tensors to remove the batch dimension of 1
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

if __name__ == '__main__':
    # Example of how to use the dataset
    dataset = SVDataset(
        tsv_path=config.PROCESSED_DATA_TSV,
        tokenizer_name=config.MODEL_NAME,
        max_length=config.MAX_TOKEN_LENGTH
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    for key, value in sample.items():
        logger.info(f"{key}: shape={value.shape}, dtype={value.dtype}")