import pandas as pd

from torch.utils.data import Dataset
from Tokenizer import tokenize_text


class TextDataset(Dataset):

    def __init__(self, train=True, max_len=100):
        path = f'./data/{"train" if train else "test"}.tsv'
        self.data = pd.read_csv(path)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = tokenize_text(self.data.iloc[idx, 0])
        attn_msk = self.attn_msk(text)
        text = self.pad(text)
        label = self.data.iloc[idx, 1]
        return text, attn_msk, label

    def pad(self, text):
        return [text[i] if i < len(text) else '<pad>' for i in range(self.max_len)]

    def attn_msk(self, text):
        return [1 if i < len(text) else 0 for i in range(self.max_len)]

