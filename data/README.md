# Data Folder

- `train.tsv` and `test.tsv` are the base train and test data.
- `neq_train.tsv` and `neq_test.tsv` are the base train and test data concatenated with the additional data we gathered and classified with the rule based model.
- All of the files beginning with `word2vec` are the pickled word2vec embeddings trained on materials papers [by the Olivetti Group](https://github.com/olivettigroup/materials-word-embeddings).
- The `/raw` directory contains all of the raw sentences without explicit labels. They are labelled by the titles of the text files.
