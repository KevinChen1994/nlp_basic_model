# -*- coding:utf-8 -*-
# author:chenmeng
# datetime:2020/3/25 17:21
# software: PyCharm
import os
import random

import jieba
import numpy as np
from tqdm import tqdm


def data_split(data_dir, data_out):
    with open(data_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        train_set = []
        test_set = []
        val_set = []
        for i, line in enumerate(lines):
            if i < 2000:
                train_set.append(line)
            elif 2000 <= i < 2500:
                test_set.append(line)
            elif 2500 <= i < 2700:
                val_set.append(line)
            else:
                break
    with open(data_out + '/train.txt', 'w', encoding='utf-8') as f:
        for i in train_set:
            f.write(i)
    with open(data_out + '/test.txt', 'w', encoding='utf-8') as f:
        for i in test_set:
            f.write(i)
    with open(data_out + '/val.txt', 'w', encoding='utf-8') as f:
        for i in val_set:
            f.write(i)


def data_analysis(data_dir):
    with open(data_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        len_ = []
        dict_ = {}
        for line in lines:
            label = line.split('\t')[0]
            if label not in dict_:
                dict_[label] = 1
            else:
                dict_[label] += 1
            content = line.split('\t')[1]
            len_.append(len(content))
        temp = np.array(len_)
        print(dict_)
        print('AVG:', np.average(temp))
        print('MEDIAN:', np.median(temp))
        print('MAX:', np.max(temp))
        print('MIN:', np.min(temp))


def data_cut(data_dir, out_dir):
    cut_list = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            content = line.split('\t')[1]
            content_ = jieba.cut(content)
            content_cut = " ".join(content_)
            cut_list.append(content_cut)
    with open(out_dir, 'w', encoding='utf-8') as f:
        for data in cut_list:
            f.write(data)


def people_format(source_dir, target_dir, out_dir):
    source_list = open(source_dir, 'r', encoding='utf-8').readlines()
    target_list = open(target_dir, 'r', encoding='utf-8').readlines()
    final_list = []
    for i in tqdm(range(len(source_list))):
        single_list = []
        sentence = source_list[i].replace('\n', '').split(' ')
        tag = target_list[i].replace('\n', '').split(' ')
        for char, label in zip(sentence, tag):
            if char != '':
                single_list.append(char + '\t' + label.replace('_', '-') + '\n')
        final_list.append(single_list)
    with open(out_dir, 'w', encoding='utf-8') as f:
        for data in tqdm(final_list):
            for line in data:
                f.write(line)
            f.write('\n')


def people_data_split(data_dir, out_dir):
    data_list = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        single_data = []
        for line in lines:
            if line == '\n':
                data_list.append(single_data)
                single_data = []
            else:
                single_data.append(line)
    random.shuffle(data_list)
    train = []
    test = []
    dev = []
    n = int(len(data_list) * 0.1)
    for i in range(n):
        if i <= 0.7 * n:
            train.append(data_list[i])
        elif 0.7 * n < i < 0.9 * n:
            test.append(data_list[i])
        else:
            dev.append(data_list[i])
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for data in train:
            for line in data:
                f.write(line)
            f.write('\n')
    with open(os.path.join(out_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        for data in test:
            for line in data:
                f.write(line)
            f.write('\n')
    with open(os.path.join(out_dir, 'dev.txt'), 'w', encoding='utf-8') as f:
        for data in dev:
            for line in data:
                f.write(line)
            f.write('\n')


if __name__ == '__main__':
    # data_split('./cnews/cnews.train.txt', './small_cnews/')
    # data_analysis('./small_cnews/train.txt')
    # data_cut('./cnews/cnews.train.txt', './cnews/cnews.cut.txt')
    people_format('人民日报2014NER数据/source_BIO_2014_cropus.txt', '人民日报2014NER数据/target_BIO_2014_cropus.txt',
                  '人民日报2014NER数据/all_data.txt')
    people_data_split('人民日报2014NER数据/all_data.txt', 'small_ner')
