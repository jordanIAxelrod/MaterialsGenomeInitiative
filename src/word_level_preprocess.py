"""
Functions that conduct word level preprocessing.
"""



import re
import collections
from typing import List



def put_int_together(w: str) -> str:
    """
    Transform an integer number expressed as a string into '<int>'.
    e.g.
    '60' -> '<int>'

    args:
      - w: a string (word)

    returns:
      - w or '<int>'
    """

    if re.fullmatch(r"[0-9]+", w) is None:
        return w
    else:
        return '<int>'



def put_decimal_together(w: str) -> str:
    """
    Transform a decimal expressed as a string into '<dec>'.
    e.g.
    '0.5' -> '<dec>'

    args:
      - w: a string (word)

    returns:
      - w or '<dec>'
    """

    """
    if re.fullmatch(r"[0-9]+(\.[0-9]+)?:[0-9]+(\.[0-9]+)?", w) is None:
        return w
    else:
        return '<dec>'
    """
    return re.sub(r"[0-9]+\.[0-9]+", "<dec>", w)



def c2temp(w: str) -> str:
    """
    Transform various expressions of degrees celsius into '<temp>'.
    e.g.
    '°c' -> '<temp>'

    args:
      - w: a string (word)

    returns:
      - w or '<temp>'
    """

    if w in ['°c', 'oc', 'c', '◦c']:
        return '<temp>'
    else:
        return w



def split_slash(w: str) -> str:
    """
    Split unit expression into parts.
    e.g.
    'g/mol' -> 'g / mol'
    'and/or' -> 'and / or'

    args:
      - w: a string (word)

    returns:
      - a string
    """

    subs = w.split('/')
    l = []
    for sub in subs:
        l.append(sub)
        l.append('/')
    l.pop(-1)
    return " ".join(l)
