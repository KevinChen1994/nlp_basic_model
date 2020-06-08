# !usr/bin/env python
# -*- coding:utf-8 _*-
# author:chenmeng
# datetime:2020/5/29 15:22
import tensorflow.contrib.keras as kr
import numpy as np


def read_corpus(corpus_dir):
    content, label = [], []
    with open(corpus_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line != '\n':
                l, c = line.strip().split('\t')
                content.append(c)
                label.append(l)
    return content, label


def vocab_build(corpus_path, vocab_path, min_count):
    word2id = {}
    content, label = read_corpus(corpus_path)
    for sentence in content:
        for word in sentence:
            if word not in word2id:
                word2id[word] = 1
            else:
                word2id[word] += 1
    # 按出现频率排序
    word2id = sorted(word2id.items(), key=lambda x: x[1], reverse=True)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write('<UNK>\n')
        f.write('<PAD>\n')
        count = 0
        for word in word2id:
            if word[1] < min_count: break
            f.write(word[0] + '\n')
            count += 1
    print(count)


def read_vocab(vocab_dir):
    with open(vocab_dir, 'r', encoding='utf-8') as f:
        words = [_.strip() for _ in f.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_labels(label_dir):
    with open(label_dir, 'r', encoding='utf-8') as f:
        labels = [_.strip() for _ in f.readlines()]
    label_to_id = dict(zip(labels, range(len(labels))))
    id_to_label = dict(zip(label_to_id.values(), label_to_id.keys()))
    return labels, label_to_id, id_to_label


def process_file(file_dir, word_to_id, cat_to_id, seq_length=128):
    contents, labels = read_corpus(file_dir)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, seq_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))

    return x_pad, y_pad

# 返回值添加了序列长度
def process_file_(file_dir, word_to_id, cat_to_id, seq_length=128):
    contents, labels = read_corpus(file_dir)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, seq_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    seq_lens = np.zeros([len(data_id)], dtype=np.int)
    for i in range(len(data_id)):
        seq_lens[i] = min(len(data_id[i]), seq_length)

    return x_pad, y_pad, seq_lens


def batch_iter(x, y, batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    # 生成一个随机排列的数组，不改变原矩阵的值
    indices = np.random.permutation(np.arange(data_len))
    # 这样处理的好处是让X和Y中的值依然保持原来的对应关系
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


# 返回值添加了序列长度
def batch_iter_(x, y, seq_lens, batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    # 生成一个随机排列的数组，不改变原矩阵的值
    indices = np.random.permutation(np.arange(data_len))
    # 这样处理的好处是让X和Y中的值依然保持原来的对应关系
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    len_shuffle = seq_lens[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id], len_shuffle[start_id:end_id]


if __name__ == '__main__':
    # vocab_build('../../data/small_cnews/test.txt', '../../data/vocab.txt', 2)
    # read_vocab('../../data/vocab.txt')
    read_labels('../../data/small_cnews/label.txt')
