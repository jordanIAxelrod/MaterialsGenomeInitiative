from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import os
class Embeddings:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.full_data = pd.concat([train_data, test_data]).apply(lambda x: ' '.join([str(y) for y in x]))
        self.bow = self.create_BOW()
        self.tfidf = self.create_tfidf()
        self.w2v = self.create_w2v()

    def create_BOW(self):
        vectorizor = CountVectorizer()
        vectorizor.fit(self.full_data)
        return vectorizor

    def BOW(self):
        """
        Returns bag of words embedding of the sentence
        :return:
        """
        return self.bow

    def create_tfidf(self):
        vectorizor = TfidfVectorizer()
        vectorizor.fit(self.full_data)
        return vectorizor

    def TFIDF(self):
        """
        Returns TFIDF representation of the sentences
        :return:
        """
        return self.tfidf

    def create_w2v(self):
        main_dir = os.path.dirname(__file__)
        w2v_path = os.path.join(main_dir, '../data/word2vec_embeddings-SNAPSHOT.model')
        w2v = Word2VecVectorizor(w2v_path)
        return w2v

    def word2vec(self):
        """
        Returns word2vec representation of the sentences
        :return:
        """
        return self.w2v


class Word2VecVectorizor:
    def __init__(self, path):
        self.word2vec = Word2Vec.load(path)

    def transform(self, text):
        w2v = []
        for sentence in text:
            s = None
            for word in sentence.split():
                if word in self.word2vec.wv:
                    if s is not None:
                        s += np.array(self.word2vec.wv[word])
                    else:
                        s = np.array(self.word2vec.wv[word])
            w2v.append(s)
        return np.array(w2v)
