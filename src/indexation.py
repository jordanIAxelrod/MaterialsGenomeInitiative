"""
Functions that conduct indexation of words.
"""



from collections import Counter
from typing import List



def word2idx(sentences: List[List[str]]) -> (List[List[int]], int):
    """
    Transform words/categories to indices.

    args:
      - sentences: a list of sentences (tokenized)

    returns:
      - indices: a list of sentences (word indices)
      - vocab_size: an integer
    """

    word_pool = [w for s in sentences for w in s]

    word_counts = sorted(
        Counter(word_pool).items(), key=lambda x: x[1], reverse=True
        )
    word_types = [x[0] for x in word_counts]
    word_type2idx = {wordtype: i for i, wordtype in enumerate(word_types)}

    indices = [[word_type2idx[w] for w in s] for s in sentences]

    return (indices, len(word_type2idx))
