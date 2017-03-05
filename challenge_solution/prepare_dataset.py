#!/usr/bin/env python3
# -*- coding: utf-8 -*-
count = 0

import pandas as pd

answer_to_label_mapping = {}
label_to_answer_mapping = {}


def get_dataset_dataframe(dataset_file_path):
    """

    :rtype: DataFrame
    """
    training_datalist = []
    with open('%s' % dataset_file_path, 'r') as training_dataset1:
        single_row = []
        count = 0
        for sentence in training_dataset1:
            if not count % 2:
                single_row.append(sentence.strip())
            else:
                single_row.append(sentence.strip())
            if not len(single_row) % 2:
                training_datalist.append(single_row)
                single_row = []
            count += 1
    df = pd.DataFrame(training_datalist, columns=['question', 'answer'])
    return df


def get_answer_label_mapping(df):
    answer_set = sorted(set(df.answer.tolist()))
    global answer_to_label_mapping
    for index, answer in enumerate(answer_set):
        answer_to_label_mapping[answer] = index
        label_to_answer_mapping[index] = answer
    return answer_to_label_mapping, label_to_answer_mapping
