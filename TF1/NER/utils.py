# !usr/bin/env python
# -*- coding:utf-8 _*-
# author:chenmeng
# datetime:2020/8/11 15:04
import numpy as np
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


def read_corpus(corpus_path):
    data = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sentence, tag = [], []
        for line in lines:
            if line != '\n':
                [char, label] = line.strip().split()
                sentence.append(char)
                tag.append(label)
            else:
                data.append((sentence, tag))
                sentence, tag = [], []
    return data


def data_process(path, w2i_char, w2i_bio, max_seq_len):
    data = read_corpus(path)
    data_id, label_id, input_mask, input_segment = [], [], [], []
    for i, (sentence, label) in enumerate(data):
        # 将sentence的tag转成id，并处理长度：大于max_seq_len的只保留到最大长度，小于的暂时不处理
        data_id.append([w2i_char[char] if char in w2i_char else w2i_char['[UNK]'] for char in sentence])
        label_id.append([w2i_bio[tag] for tag in label])
        data_id[i] = data_id[i][:max_seq_len] if len(data_id[i]) > max_seq_len else data_id[i]
        label_id[i] = label_id[i][:max_seq_len] if len(label_id[i]) > max_seq_len else label_id[i]
        # 开头添加[CLS]，结尾添加[SEP]
        data_id[i].insert(0, w2i_char['[CLS]'])
        data_id[i].append(w2i_char['[SEP]'])
        label_id[i].insert(0, w2i_bio['O'])
        label_id[i].append(w2i_bio['O'])
        input_mask.append([1] * len(data_id[i]))
        input_segment.append([0] * len(data_id[i]))
        # 处理不够长的句子，如果句子长度小于max_seq_len，需要填充['PAD']，因为添加了['CLS']和['SEP']，所以长度需要加2
        input_mask[i].extend([0] * (max_seq_len - len(data_id[i]) + 2))
        input_segment[i].extend([0] * (max_seq_len - len(data_id[i]) + 2))
        data_id[i].extend([w2i_char['[PAD]']] * (max_seq_len - len(data_id[i]) + 2))
        label_id[i].extend([w2i_bio['O']] * (max_seq_len - len(label_id[i]) + 2))

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
    data_process('../../data/small_ner/train.txt', '', '', '')
