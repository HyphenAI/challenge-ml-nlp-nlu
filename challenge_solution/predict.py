#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

from challenge_solution.prepare_dataset import label_to_answer_mapping
from challenge_solution.training_model import train_using_bag_word_model


def main():
    model, vectorizor = train_using_bag_word_model()
    print('to exit , enter 0')
    while True:
        input_str = input('Enter Question\n').strip()
        if input_str == '0':
            break
        y_pred = model.predict(vectorizor.transform([input_str]).toarray())
        print(y_pred)
        for pred in y_pred:
            print('Predicted Answer : ', label_to_answer_mapping.get(pred))




if __name__ == '__main__':
    sys.exit(main())
