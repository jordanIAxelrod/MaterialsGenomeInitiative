"""
Functions that conduct sentence level preprocessing, which requires information of more than unigrams.
"""

import re
from typing import List, Tuple
from word_level_preprocess import *
from indexation import *
import nltk
import itertools

numpy_records = List[Tuple]



def rm_linebreaks(text: str, p=False) -> str:
    """
    Removes line breaks.
    e.g.
    "transmission elec- tron microscopy"
    -> "transmission electron microscopy"

    args:
      - text: a string
      - p: a boolean for whether or not to print texts with line breaks

    returns:
      - text_dh: a string dehyphenated
    """
    text_dh = re.sub(r"([a-z])- ([a-z])", r"\1\2", text)

    if p and text != text_dh:
        print("before:", text)
        print("after :", text_dh)

    return text_dh



def c2temp_2(text: str) -> str:
    """
    Transform two-words expressions of degrees celsius into '<temp>'.
    e.g.
    '° c' -> '<temp>'

    args:
      - text: a string

    returns:
      -
    """

    text_t = re.sub(r"° c", '<temp>', text)
    text_t = re.sub(r"◦ c", '<temp>', text)
    return text_t

def preprocess(data: numpy_records) -> numpy_records:
    # remove line breaks (e.g. "elec- tron" -> "electron")
    data = [(rm_linebreaks(t), l) for t, l in data]

    # lower case
    data = [(t.lower(), l) for t, l in data]

    # unify expressions for temperature (e.g. '° c' -> '<temp>')
    data = [(c2temp_2(t), l) for t, l in data]

    data = [(nltk.tokenize.word_tokenize(t), l) for t, l in data]

    # recognize integer as '<int>' (e.g. '60' -> '<int>')
    data = [([put_int_together(w) for w in t], l) for t, l in data]

    # recognize decimal as '<dec>' (e.g. '0.5' -> '<dec>')
    data = [([put_decimal_together(w) for w in t], l) for t, l in data]

    # recognize ratioas '<ratio>' (e.g. '1:1' -> '<ratio>')
    data = [([put_ratio_together(w) for w in t], l) for t, l in data]

    # split slash (e.g. 'g/mol' -> '['g', '/', 'mol'])
    data = [([split_slash(w) for w in t], l) for t, l in data]
    data = [(list(itertools.chain.from_iterable(t)), l) for t, l in data] # flatten

    # unify expressions for temperature (e.g. '°c' -> '<temp>')
    data = [([c2temp(w) for w in t], l) for t, l in data]
    
    return data

def index_words(data: numpy_records):
    texts   = [x[0] for x in data]
    targets = [[x[1]] for x in data]
    texts, vocab_size, _   = word2idx(texts)
    targets, _, _          = word2idx(targets)
    targets = [l[0] for l in targets]
    data_idx = list(zip(texts, targets))
    return (data_idx, vocab_size, targets)