#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nltk


def remove_stopwords(sentence):
    pass


def word_tokenizer(sentence):
    tokens = nltk.word_tokenize(sentence)
    return tokens


def sentence_tokenizer(text):
    sentence_list = nltk.sent_tokenize(text)
    return sentence_list
