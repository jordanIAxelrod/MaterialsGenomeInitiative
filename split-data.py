import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List

CATEGORIES = ['action', 'constituent', 'null', 'property']

def make_df_from_txt(filepath: str, label: str) -> pd.DataFrame:
    with open(filepath) as f:
        texts = [{'label': label, 'text': line.rstrip()} for line in f]
    return pd.DataFrame(texts, columns=['label','text'])

def tt_split_df(df: pd.DataFrame) -> Tuple[pd.DataFrame]:
    # train-test split done on each category to ensure both sets have same ratio of all labels
    y = df.pop('label')
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=11122022) # set random state keeps this the same for everyone
    X_train['label'] = y_train
    X_test['label'] = y_test
    return (X_train, X_test)

trains = []
tests = []
for name in CATEGORIES:
    df = make_df_from_txt(f'./data/raw/{name}.txt', name)
    tr, ts = tt_split_df(df)
    trains.append(tr)
    tests.append(ts)

pd.concat(trains).to_csv('./data/train.tsv', sep='\t', index=False)
pd.concat(tests).to_csv('./data/test.tsv', sep='\t', index=False)