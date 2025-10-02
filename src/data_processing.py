import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

def load_and_clean_data(file_path):
    """
    Loads a CSV file into a pandas DataFrame and performs essential cleaning.
    - Drops rows with missing 'cleaned_text' or 'label'.
    - Ensures 'cleaned_text' is a string.
    """
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['cleaned_text', 'label'], inplace=True)
        df['cleaned_text'] = df['cleaned_text'].astype(str)
        return df
    except FileNotFoundError:
        print(f"Error: The dataset file was not found at {file_path}.")
        print("Please ensure your CSV files are in the 'data/' directory.")
        return None

class HateSpeechDataset(Dataset):
    """
    Custom PyTorch Dataset for tokenizing and serving hate speech text data.
    """
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True, # Required for some models like DeBERTa
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'token_type_ids': inputs['token_type_ids'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size, shuffle=False):
    """
    Creates a PyTorch DataLoader for a given DataFrame.
    """
    ds = HateSpeechDataset(
        texts=df.cleaned_text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

