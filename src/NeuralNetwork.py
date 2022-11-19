import torch
import torch.nn as nn


class RNN(nn.Module):

    def __init__(self, num_classes: int, hidden_dim: int, num_words: int, embed_dim: int, to_embed: bool):
        super().__init__()
        self.to_embed = to_embed
        if to_embed:
            self.embed = nn.Embedding(num_words, embed_dim)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            1,
            dropout=.1,
            bidirectional=True,
            batch_first=True
        )
        self.mha = nn.MultiheadAttention(hidden_dim, 1, .1, batch_first=True)

        self.proj = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, attn_msk):
        """
        Forward pass of the GRU Attention network
        :param X: the sentences
        :return:
        """

        if self.to_embed:
            X = self.embed(X)

        X = self.gru(X)
        X = self.mha(X, X, X, key_padding_mask=attn_msk)
        X = self.proj(X)
        return self.softmax(X)


class NNTrainer:
    def __init__(self, nn, epochs, optimizer, loss):
        self.nn = nn
        self.epochs = epochs
        self.optimizer = optimizer(nn.parameters())
        self.loss = loss

    def fit(self, X, Y):

        for i in self.epochs:
            self.nn.train()
            for j, (x, y) in enumerate(zip(X, Y)):
                self.optimizer.zero_grad()
                pred = self.nn(x)
                loss = self.loss(pred, y)
                self.optimizer.step()

    def predict(self, data):
        return self.nn(data)
