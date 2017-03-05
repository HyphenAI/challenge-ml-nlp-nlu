#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

from challenge_solution.feature_generation import CorpusFeatureVector
from challenge_solution.model import BasicClassifier
from challenge_solution.prepare_dataset import get_dataset_dataframe, get_answer_label_mapping

vectorizer = CountVectorizer()

training_dataset_file_2 = 'training_dataset_2.txt'
testing_dataset_file_2 = 'test-data.txt'


def transform_data_to_bow_feature_vectors(vectorizer, training_dataset_dataframe, answer_label_mapping):
    X, Y = [], []
    X = vectorizer.transform(training_dataset_dataframe.question.tolist()).toarray()
    for _, row in training_dataset_dataframe.iterrows():
        y = answer_label_mapping.get(row.answer, -1)  # Possible that test data has answer not in training data
        Y.append(y)
    return np.array(X), np.array(Y)


def get_analyser_from_dataframe(training_dataset_dataframe):
    corpus = training_dataset_dataframe.question.tolist()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X


def get_data(training_dataset_file, testing_dataset_file):
    training_dataset_dataframe = get_dataset_dataframe(training_dataset_file)
    testing_dataset_dataframe = get_dataset_dataframe(testing_dataset_file)
    return training_dataset_dataframe, testing_dataset_dataframe


def transform_data_to_word_embedding_feature_vectors(training_dataset_dataframe, answer_label_mapping):
    corpus = training_dataset_dataframe.question.tolist()
    X = CorpusFeatureVector(corpus).transform()
    Y = []
    for _, row in training_dataset_dataframe.iterrows():
        y = answer_label_mapping.get(row.answer, -1)  # Possible that test data has answer not in training data
        Y.append(y)
    return np.array(X), np.array(Y)


def train_using_word_embedding(training_dataset_file='training_dataset.txt', testing_dataset_file='test_dataset.txt'):
    def show_predicted_vs_actual(Y_test, Y_predicted):
        dataframe_data = []
        for (_, row), y_label in zip(testing_dataset_dataframe.iterrows(), Y_predicted):
            dataframe_data.append([row.question, row.answer, label_to_answer_mapping.get(y_label)])

        import pandas as pd
        df = pd.DataFrame(dataframe_data, columns=['question', 'actual_answer', 'predicted_answer'])
        pd.set_option('display.expand_frame_repr', False)
        return df

    training_dataset_dataframe, testing_dataset_dataframe = get_data(training_dataset_file, testing_dataset_file)
    answer_label_mapping, label_to_answer_mapping = get_answer_label_mapping(training_dataset_dataframe)

    X_train, Y_train = transform_data_to_word_embedding_feature_vectors(training_dataset_dataframe,
                                                                        answer_label_mapping)
    X_test, Y_test = transform_data_to_word_embedding_feature_vectors(testing_dataset_dataframe, answer_label_mapping)

    model = BasicClassifier()
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)
    report = classification_report(Y_test, Y_predicted)
    show_predicted_vs_actual(Y_test, Y_predicted)
    print('\n\nCLASSIFICATION_REPORT :\n', report)
    return model


def train_using_bag_word_model(training_dataset_file='training_dataset.txt', testing_dataset_file='test_dataset.txt'):
    def show_predicted_vs_actual(Y_test, Y_predicted):
        dataframe_data = []
        for (_, row), y_label in zip(testing_dataset_dataframe.iterrows(), Y_predicted):
            dataframe_data.append([row.question, row.answer, label_to_answer_mapping.get(y_label)])

        import pandas as pd
        df = pd.DataFrame(dataframe_data, columns=['question', 'actual_answer', 'predicted_answer'])
        pd.set_option('display.expand_frame_repr', False)
        return df

    training_dataset_dataframe, testing_dataset_dataframe = get_data(training_dataset_file, testing_dataset_file)
    answer_label_mapping, label_to_answer_mapping = get_answer_label_mapping(training_dataset_dataframe)

    corpus = training_dataset_dataframe.question.tolist()
    global vectorizer
    vectorizer.fit(corpus)

    X_train, Y_train = transform_data_to_bow_feature_vectors(vectorizer, training_dataset_dataframe,
                                                             answer_label_mapping)
    X_test, Y_test = transform_data_to_bow_feature_vectors(vectorizer, testing_dataset_dataframe, answer_label_mapping)

    model = BasicClassifier()
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)
    df = show_predicted_vs_actual(Y_test, Y_predicted)
    report = classification_report(Y_test, Y_predicted)
    print('\n\nCLASSIFICATION_REPORT :\n', report)
    return model, df


if __name__ == '__main__':
    sys.exit(train_using_bag_word_model())
