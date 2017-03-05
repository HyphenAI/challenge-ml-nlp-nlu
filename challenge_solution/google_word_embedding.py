# -*- coding: utf-8 -*-
import logging
import os
import pickle

from gensim.models import KeyedVectors as Word2Vec


class GoogleWord2Vec(object):
    FILENAME = '/opt/GoogleNews_Word2Vec.bin'
    PICKLE_FILENAME = FILENAME + '.pkl'

    def __init__(self):
        self.model = None

    def get_model(self):
        if self.model is not None:
            return self.model
        elif os.path.isfile(self.PICKLE_FILENAME):
            logging.info('Reading from the pickled GW2Vec: ' + self.PICKLE_FILENAME)
            self.model = pickle.load(open(self.PICKLE_FILENAME, 'rb'))
            return self.model
        else:
            self.model = Word2Vec.load_word2vec_format(
                self.FILENAME, binary=True
            )
            pickle.dump(self.model, open(self.PICKLE_FILENAME, 'wb'))
            return self.model
