import pandas as pd
import torch
import nltk
import re
from typing import List

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
# df = pd.read_csv('./data/train.tsv', sep='\\t')

def tokenize_text(text: str) -> List[str]:
    # Saw hyphens like this in the middle of a lot of words e.g. \"transmission elec- tron microscopy\". Seem like line breaks
    text = re.sub(r'([a-z])- ([a-z])', '\\1\\2', text)
    return nltk.tokenize.word_tokenize(text)

"pd.DataFrame([(tokenize_text(t),l) for t,l in df.to_records(index=False)], columns=df.columns)"