# !usr/bin/env python
# -*- coding:utf-8 _*-
# author:chenmeng
# datetime:2020/9/1 11:24
import pandas as pd
import numpy as np
import os
import argparse


def load_vocabulary(path):
    vocab = open(path, 'r', encoding='utf-8').read().strip().split('\n')
    print('load vocab from: {}, containing words: {}'.format(path, len(vocab)))
    w2i = {}
    i2w = {}
    for i, w in enumerate(vocab):
        w2i[w] = i
        i2w[i] = w
    return w2i, i2w


def read_labels(data_dir):
    data = pd.read_csv(data_dir)
    i2w = {}
    w2i = {}
    for item in data.itertuples():
        i2w[item[1]] = item[2]
        w2i[item[2]] = item[1]
    print('load label from: {}, containing labels: {}'.format(data_dir, len(i2w)))
    return w2i, i2w


def data_split(data_dir, data_out):
    data = pd.read_csv(data_dir)
    data_list = []
    for item in data.itertuples():
        # 前271个是英文商品
        if item[1] > 271 and type(item[2]) is not float:
            product_name = item[2].replace(' ', '')
            category_ids = item[3]
            data_list.append((product_name, category_ids))
    n = len(data_list)
    train = []
    dev = []
    test = []
    for i in range(n):
        if i < 0.7 * n:
            train.append(data_list[i])
        elif 0.7 * n <= i < 0.9 * n:
            dev.append(data_list[i])
        else:
            test.append(data_list[i])
    with open(os.path.join(data_out, 'train.txt'), 'w', encoding='utf-8') as f:
        for data_single in train:
            f.write(data_single[0] + '\t' + data_single[1] + '\n')
    with open(os.path.join(data_out, 'dev.txt'), 'w', encoding='utf-8') as f:
        for data_single in dev:
            f.write(data_single[0] + '\t' + data_single[1] + '\n')
    with open(os.path.join(data_out, 'test.txt'), 'w', encoding='utf-8') as f:
        for data_single in test:
            f.write(data_single[0] + '\t' + data_single[1] + '\n')


def read_corpus(data_dir):
    data_list = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip().split('\t')
            product_name = list(item[0])
            category_ids = item[1].split(',')
            data_list.append((product_name, category_ids))
    return data_list


def data_process(path, w2i_char, w2i_label, max_seq_len):
    data = read_corpus(path)
    data_id, label_id, input_mask, input_segment = [], [], [], []
    for i, (sentence, label) in enumerate(data):
        # 将sentence的tag转成id，并处理长度：大于max_seq_len的只保留到最大长度，小于的暂时不处理
        data_id.append([w2i_char[char] if char in w2i_char else w2i_char['[UNK]'] for char in sentence])
        label_id.append([1 if str(label_index) in label else 0 for label_index in range(len(w2i_label))])
        data_id[i] = data_id[i][:max_seq_len] if len(data_id[i]) > max_seq_len else data_id[i]
        # 开头添加[CLS]，结尾添加[SEP]
        data_id[i].insert(0, w2i_char['[CLS]'])
        data_id[i].append(w2i_char['[SEP]'])
        input_mask.append([1] * len(data_id[i]))
        input_segment.append([0] * len(data_id[i]))
        # 处理不够长的句子，如果句子长度小于max_seq_len，需要填充['PAD']，因为添加了['CLS']和['SEP']，所以长度需要加2
        input_mask[i].extend([0] * (max_seq_len - len(data_id[i]) + 2))
        input_segment[i].extend([0] * (max_seq_len - len(data_id[i]) + 2))
        data_id[i].extend([w2i_char['[PAD]']] * (max_seq_len - len(data_id[i]) + 2))

    print('Load data num: {}'.format(len(data)))
    return data_id, label_id, input_mask, input_segment


def batch_iter(seq_id, label_id, input_mask, input_segment, batch_size):
    data_len = len(seq_id)
    num_batch = int((data_len - 1) / batch_size) + 1
    # 将list转换成ndarray，方便打乱顺序
    seq_id = np.array(seq_id, dtype='int32')
    label_id = np.array(label_id, dtype='int32')
    input_mask = np.array(input_mask, dtype='int32')
    input_segment = np.array(input_segment, dtype='int32')
    # 生成随机排列的数组，不改变原矩阵的值，使四个ndarray保持对应的关系
    indices = np.random.permutation(data_len)
    seq_id_shuffle = seq_id[indices]
    label_id_shuffle = label_id[indices]
    input_mask_shuffle = input_mask[indices]
    input_segment_shuffle = input_segment[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield seq_id_shuffle[start_id: end_id], label_id_shuffle[start_id: end_id], \
              input_mask_shuffle[start_id: end_id], input_segment_shuffle[start_id: end_id]


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # read_labels('../../data/amazon/categories.csv')
    read_corpus('../../data/amazon/train.txt')
    # data_split('../../data/amazon/products.csv', '../../data/amazon/')
