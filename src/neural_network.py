import os

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
import seaborn as sns
import matplotlib.pyplot as plt
class RNN(nn.Module):

    def __init__(self, num_classes: int, hidden_dim: int, num_words: int, embed_dim: int, to_embed: bool,
                 word_list: list):
        super().__init__()
        self.to_embed = to_embed
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(num_words + 1, embed_dim)

        # If not using new embeddings, use the embeddings from
        # https://github.com/olivettigroup/materials-word-embeddings
        self.word_list = word_list

        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            1,
            bidirectional=True,
            batch_first=True
        )
        self.Uw = nn.Parameter(torch.randn(hidden_dim)).reshape(1, 1, -1)
        self.mha = nn.MultiheadAttention(hidden_dim, 1, .1, batch_first=True)

        self.proj = nn.Linear(hidden_dim, num_classes)

        self.softmax = nn.Softmax(dim=-1)

    def set_embedding_weights(self):
        main_dir = os.path.dirname(__file__)
        w2v_path = os.path.join(main_dir, '../data/word2vec_embeddings-SNAPSHOT.model')
        embedding = Word2Vec.load(w2v_path)
        weights = torch.zeros(len(self.word_list) + 1, self.embed_dim)
        for i, word in enumerate(self.word_list):
            if word in embedding.wv:
                weights[i] = torch.Tensor(np.array(embedding.wv[word]))
            else:
                weights[i] = torch.randn(self.embed_dim)
        weights[-1] = torch.randn(self.embed_dim)
        self.embed.weight = torch.nn.Parameter(weights)

    def forward(self, X, attn_msk):
        """
        Forward pass of the GRU Attention network
        :param attn_msk:
        :param X: the sentences
        :return:
        """

        X = self.embed(X)
        bsz, l, embed_dim = X.shape
        X = self.gru(X)[0]
        X = X[:, :, :self.hidden_dim] + X[:, :, self.hidden_dim:]
        X, weights = self.mha(self.Uw.repeat(bsz, 1, 1), X, X, key_padding_mask=attn_msk.bool())
        X = self.proj(X)
        X = X.mean(dim=1)
        return X, weights


class NNTrainer:
    def __init__(self, nn, epochs, optimizer, loss, train_loader, test_loader):
        self.nn = nn
        self.epochs = epochs
        self.optimizer = optimizer(nn.parameters())
        self.loss = loss
        self.train_loader = train_loader
        self.test_loader = test_loader

    def fit(self, X, Y):

        for i in range(self.epochs):
            self.nn.train()
            train_loss = 0
            for j, (x, msk, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                pred = self.nn(x, msk)[0]
                loss = self.loss(pred, y.flatten())
                loss.backward()
                train_loss += loss.detach()
                self.optimizer.step()
            test_loss = 0
            self.nn.eval()
            for j, (x, msk, y) in enumerate(self.test_loader):
                pred = self.nn(x, msk)[0]
                loss = self.loss(pred, y.flatten())
                test_loss += loss.detach()
            print(f"In epoch {i} the loss was {test_loss}")

    def predict(self, data=None):
        self.nn.eval()
        if data is None:
            pred = []

            for j, (x, attn, y) in enumerate(self.test_loader):
                pred.append(self.nn(x, attn)[0])
            pred = torch.cat(pred, dim=0)
            return pred
        return self.nn(data)

    def make_heatmap(self, sentence, msk, path):
        weights = self.nn(sentence, msk)[1]
        words = [self.nn.word_list[word] for i, word in enumerate(sentence[0]) if not msk[0, i]]
        print(words)
        print(weights)
        weights = weights.squeeze()[: len(words)].reshape(-1, len(words) // 3)
        words = [words[: len(words) // 3], words[len(words) // 3: len(words) * 2 // 3], words[len(words) * 2// 3:]]
        heat_map = sns.heatmap(weights.detach(), annot=words, fmt='')
        plt.savefig(path)
