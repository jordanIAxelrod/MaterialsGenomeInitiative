import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


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

    def run_svm(self, kernel, data):
        train_data, test_data = data
        svm = SVC(kernel=kernel)
        svm.fit(train_data, self.train_data['label'])
        test_pred = svm.predict(test_data)
        return test_pred

    def run_lr(self, data):
        train_data, test_data = data

        lr = LogisticRegression(max_iter=300)
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

    def embed(self, embedding):
        return embedding.transform(
            self.train_data.text.apply(lambda x: ' '.join([str(y) for y in x]))
        ), embedding.transform(
            self.test_data.text.apply(lambda x: ' '.join([str(y) for y in x]))
        )


