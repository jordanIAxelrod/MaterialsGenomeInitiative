import torch
import torch.nn as nn
from gensim.models import Word2Vec


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
            dropout=.1,
            bidirectional=True,
            batch_first=True
        )
        self.Uw = nn.Parameter(torch.randn(hidden_dim))
        self.mha = nn.MultiheadAttention(hidden_dim, 1, .1, batch_first=True)

        self.proj = nn.Linear(hidden_dim, num_classes)

        self.softmax = nn.Softmax(dim=-1)

    def set_embedding_weights(self):
        embedding = Word2Vec.load('../data/word2vec_embeddings-SNAPSHOT.model')
        weights = torch.zeros(len(self.word_list), self.embed_dim)
        for i, word in enumerate(self.word_list):
            if word in embedding.wv:
                weights[i] = torch.Tensor(embedding.wv[word])
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

        X = self.gru(X)[0]
        X = X[:, :, :self.hidden_dim] + X[:, :, self.hidden_dim:]
        X = self.mha(self.Uw, X, X, key_padding_mask=attn_msk)[0]

        X = self.proj(X)
        X = X.mean(dim=1)
        return X


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

                pred = self.nn(x, msk)
                loss = self.loss(pred, y.flatten())
                loss.backward()
                train_loss += loss.detach()
                self.optimizer.step()
            test_loss = 0
            self.nn.eval()
            for j, (x, msk, y) in enumerate(self.test_loader):
                pred = self.nn(x, msk)
                loss = self.loss(pred, y.flatten())
                test_loss += loss.detach()
            print(f"In epoch {i} the loss was {test_loss}")



    def predict(self, data=None):
        self.nn.eval()
        if data is None:
            pred = []

            for j, (x, attn, y) in enumerate(self.test_loader):
                pred.append(self.nn(x, attn))
            pred = torch.cat(pred, dim=0)
            return pred
        return self.nn(data)
