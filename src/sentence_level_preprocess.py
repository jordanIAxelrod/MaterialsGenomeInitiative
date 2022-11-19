"""
Functions that conduct sentence level preprocessing, which requires information of more than unigrams.
"""



import re



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
