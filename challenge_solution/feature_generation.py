#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nltk import PorterStemmer

from challenge_solution.nlp_utils import word_tokenizer

stemmer = PorterStemmer()
word_embedding_model = {}


class SentenceFeatureVector:
    """
    Alternate to Bag of Word model
    Could use Google Word to Vec instead a random vec generator

    """

    def __init__(self, sentence):
        self.sentence = sentence
        global word_embedding_model
        self.word_embedding_model = word_embedding_model

    def get_feature_vector(self):
        list_of_words = word_tokenizer(self.sentence)
        similar_matrix = []
        for word in list_of_words:
            w = stemmer.stem(word)
            vector = word_embedding_model.get(w, None)
            if vector is not None:
                similar_matrix.append(vector)
            else:
                vector = np.random.rand(300)
                word_embedding_model[w] = vector
                similar_matrix.append(vector)

        return np.sum(similar_matrix, axis=0)


class CorpusFeatureVector:
    def __init__(self, corpus):
        self.corpus = corpus

    def transform(self):
        X = []
        for sentence in self.corpus:
            fv = SentenceFeatureVector(sentence).get_feature_vector()
            X.append(fv)
        return np.array(X)
