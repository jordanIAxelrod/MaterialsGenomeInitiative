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



def put_ratio_together(w: str) -> str:
    """
    Transform a ratio expressed as a string into '<ratio>'.
    e.g.
    '1:1' -> '<ratio>'

    args:
      - w: a string (word)

    returns:
      - w or '<ratio>'
    """

    """
    if re.fullmatch(r"[0-9]+:[0-9]+", w) is None:
        return w
    else:
        return '<ratio>'
    """
    return re.sub(r"<(int|dec)> : <(int|dec)>", "<ratio>", w)



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



def split_with_keys(s: str, keys: str) -> str:
    """
    Split with a key.
    e.g.
    When key='(', '(1:1)' -> '( 1:1 )'
    When key='.', 'done.' -> 'done .'

    args:
      - s: a string (sentence)

    returns:
      - a string
    """
    for k in keys:
        s = s.replace(k, ' '+k+' ').replace('  ', ' ').strip()
    return s
