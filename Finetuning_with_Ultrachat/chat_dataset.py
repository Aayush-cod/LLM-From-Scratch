import torch
from torch.utils.data import Dataset
import json

class ChatDataset(Dataset):
    def __init__(self, json_path, tokenizer, context_length=384):
        self.examples = []
        self.context_length = context_length
        self.tokenizer = tokenizer

        with open(json_path, "r") as f:
            data = json.load(f)

        for item in data:
            text = item["text"]
            token_ids = tokenizer.encode(text)

            if len(token_ids) > context_length:
                token_ids = token_ids[:context_length]

            self.examples.append(torch.tensor(token_ids, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]