import pandas as pd
import torch
import numpy as np

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from scipy.stats import uniform
from scipy.sparse._csr import csr_matrix


SVM_GRID = {
    'C': uniform(1e-4, 1e4),
    'gamma': uniform(1e-5, 1),
    'class_weight': [None, 'balanced']
}

LR_GRID = {
    'C': uniform(0.001, 1000),
     'class_weight': [None, 'balanced']
}


class PaperReplicator:
    """
    Given a list of models and the data, run the data through each model and return the results
    """

    def __init__(self, train_data, test_data, neural_network_e, neural_network_w):
        self.train_data = train_data
        self.test_data = test_data
        self.nn = neural_network_e
        self.nn_2 = neural_network_w

    def create_results(self, experiments):
        experiments['F1'] = 0
        for _, experiment in experiments.iterrows():
            if not isinstance(experiment.vectorizor, bool):
                train_data, test_data = self.embed(experiment.vectorizor)

            if experiment.kernel is not None:
                print('svm')
                test_pred = self.run_svm(experiment.kernel, (train_data, test_data))
            elif experiment['name'] == 'nn':
                print('nn')
                if experiment.vectorizor:
                    test_pred = self.run_NN(1,(train_data, test_data))
                else:
                    test_pred = self.run_NN(2 , (train_data, test_data))

            else:
                print('lr')
                test_pred = self.run_lr((train_data, test_data))
            experiments.loc[_, 'F1'] = f1_score(self.test_data.label, test_pred, average='micro')
            print(experiments.loc[_, :])
        return experiments

    def tune_model(self, model, grid, X, y): 
        model_grid = RandomizedSearchCV(model, grid, refit=True, n_jobs=-1, scoring='f1_micro')
        model_grid.fit(X, y)
        return model_grid

    def run_svm(self, kernel, data):
        train_data, test_data = data
        svm = SVC(kernel=kernel)
        # tuned_svm = self.tune_model(svm, SVM_GRID, train_data, self.train_data['label'])
        svm.fit(train_data, self.train_data['label'])
        test_pred = svm.predict(test_data)
        return test_pred

    def run_lr(self, data):
        train_data, test_data = data

        lr = LogisticRegression(max_iter=3000)
        # tuned_lr = self.tune_model(lr, LR_GRID, train_data, self.train_data['label'])
        lr.fit(train_data, self.train_data['label'])
        test_pred = lr.predict(test_data)
        return test_pred

    def run_NN(self, num, data):
        train_data, test_data = data
        if num == 1:
            self.nn_2.fit(train_data, self.train_data['label'])
            test_pred =self.nn_2.predict()
        else:
            self.nn.fit(train_data, self.train_data['label'])
            test_pred = self.nn.predict()

        test_pred = list(torch.argmax(test_pred, dim=-1).flatten())
        labels = [
            'action',
            'constituent',
            'unrelated',
            'property'
        ]
        test_pred = [labels[x.item()] for x in test_pred]

        return test_pred

    def build_scalar(self, train, test):
        total = np.concatenate((train, test))
        return StandardScaler().fit(total)

    def embed_set(self, embedding, X):
        embedded_text = embedding.transform(X.text.apply(lambda x: ' '.join([str(y) for y in x])))
        if type(embedded_text) == csr_matrix:
            embedded_text = np.asarray(embedded_text.todense())
        return embedded_text

    def embed(self, embedding):
        embedded_train = self.embed_set(embedding, self.train_data)
        embedded_test = self.embed_set(embedding, self.test_data)
        scalar = self.build_scalar(embedded_train, embedded_test)
        return  (scalar.transform(embedded_train), scalar.transform(embedded_test))


