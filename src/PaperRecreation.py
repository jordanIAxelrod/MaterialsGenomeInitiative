import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class PaperReplicator:
    """
    Given a list of models and the data, run the data through each model and return the results
    """

    def __init__(self, train_data, test_data, models_to_recreate: list, neural_network):
        self.train_data = train_data
        self.test_data = test_data
        self.models_to_recreate = models_to_recreate
        self.nn = neural_network

    def create_results(self, experiments):
        experiments['F1'] = 0
        for _, experiment in experiments.iterrows():
            train_data, test_data = self.embed(experiment.vectorizor)
            if experiment.kernel is not None:
                test_pred = self.run_svm(experiment.kernel, (train_data, test_data))
            elif experiment.name == 'nn':
                test_pred = self.run_NN((train_data, test_data))
            else:
                test_pred = self.run_lr((train_data, test_data))
            experiment.F1 = f1_score(self.test_data.label, test_pred)
        return experiments

    def run_svm(self, kernel, data):
        train_data, test_data = data
        svm = SVC(kernel=kernel)
        svm.fit(train_data, self.train_data['label'])
        test_pred = svm.predict(test_data)
        return test_pred

    def run_lr(self, data):
        train_data, test_data = data

        lr = LogisticRegression()
        lr.fit(train_data, self.train_data['label'])
        test_pred = lr.predict(test_data)
        return test_pred

    def run_NN(self, data):
        train_data, test_data = data

        self.nn.fit(train_data, self.train_data['label'])
        test_pred = self.nn.predict(test_data)
        return test_pred

    def embed(self, embedding):
        return embedding.transform(self.train_data.text), embedding.transform(self.test_data.text)


