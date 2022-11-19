"""
Functions that turn index data into features to be fed into classifier models.
"""



from collections import Counter



def bow(word_ids: list, vocab_size: int) -> list:
    """
    Bag of words.
    e.g.
    [0, 0, 3]
    -> [2, 0, 0, 1]

    args:
      - word_ids: a list of word indices
      - vocab_size: an integer that determines the size of the feature vector

    returns:
      - a list of integers, each element of which is count of word indices
    """

    counts = Counter(word_ids)
    return [counts[i] for i in range(vocab_size)]
