import pandas as pd

import torch
from torch.utils.data import Dataset
from Tokenizer import tokenize_text


class TextDataset(Dataset):

    def __init__(self, df, max_len=100, num_words: int = 100):
        self.num_words = num_words
        self.data = df.copy()
        self.data['text'] = self.data.text.apply(lambda x: ' '.join(str(y) for y in x))
        labels = {
            'action': 0,
            'constituent': 1,
            'unrelated': 2,
            'property': 3
        }
        self.data['label'] = self.data['label'].map(labels)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 2]

        attn_msk = torch.LongTensor(self.attn_msk(text))
        text = torch.LongTensor(self.pad(text))
        label = torch.LongTensor([self.data.iloc[idx, 1]])
        return text, attn_msk, label

    def pad(self, text):
        return [int(text[i]) if i < len(text) else self.num_words for i in range(self.max_len)]

    def attn_msk(self, text):
        return [1 if i < len(text) else 0 for i in range(self.max_len)]
