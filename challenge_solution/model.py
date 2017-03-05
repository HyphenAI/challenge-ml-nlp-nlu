#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.naive_bayes import GaussianNB


class BasicClassifier(object):
    """
    Experimenting with basic classifier.
    """

    def __init__(self):
        self.model = GaussianNB()

    def get_model(self):
        return self.model

    def fit(self, X, Y):
        """

        :rtype: object
        """
        self.model.fit(X, Y)

    def predict(self, X):
        """

        :rtype: object
        """
        assert self.model is not None
        return self.model.predict(X)


class LSTM:
    """
    Implement only when Basic Classifier fail to perform minimum requirements.
    Keep the same interface

    """

    def get_model(self):
        pass

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass


class Model(BasicClassifier):
    def __init__(self):
        super().__init__()
