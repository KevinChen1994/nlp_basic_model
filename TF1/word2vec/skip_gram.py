# !usr/bin/env python
# -*- coding:utf-8 _*-
# author:chenmeng
# datetime:2020/6/23 10:51

import time
import numpy as np
import tensorflow as tf
import random
from collections import Counter
from tqdm import tqdm

'''数据预处理'''
with open('../../data/cnews/cnews.cut.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    words = []
    for line in tqdm(lines):
        line_split = line.split(' ')
        for l in line_split:
            words.append(l)
words_count = Counter(words)
# 筛选低频词
words = [w for w in words if words_count[w] > 50]

vocab = set(words)
word_to_id = {w: i for i, w in enumerate(vocab)}
id_to_word = {i: w for i, w in enumerate(vocab)}

print('total words:{}'.format(len(words)))
print('unique words:{}'.format(len(set(words))))

# 对原文本进行词到id的转换
id_words = [word_to_id[w] for w in words]

'''
负采样，去除一些停用词之类的词，加快训练速度，同时减少训练过程中的噪音
我们采用以下公式: $$ P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}} $$
 
其中$ t $是一个阈值参数，一般为1e-3至1e-5。
$f(w_i)$ 是单词 $w_i$ 在整个数据集中的出现频次。
$P(w_i)$ 是单词被删除的概率。
'''
t = 1e-5
threshold = 0.9
# 统计单词出现频次
int_word_counts = Counter(id_words)
total_count = len(id_words)
# 计算单词出现的频率
word_freqs = {w: c / total_count for w, c in int_word_counts.items()}
# 计算被删除的概率
prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
# 对单词进行采样
train_words = [w for w in id_words if prob_drop[w] < threshold]

print('train words:{}'.format(len(train_words)))

'''
对于一个给定词，离它越近的词可能与它越相关，离它越远的词越不相关，这里我们设置窗口大小为5，对于每个训练单词，我们还会在[1:5]之间随机生成一个整数R，
用R作为我们最终选择output word的窗口大小。这里之所以多加了一步随机数的窗口重新选择步骤，是为了能够让模型更聚焦于当前input word的邻近词。
'''


def get_targets(words, idx, window_size=5):
    '''
    获得input word的上下文单词列表

    参数
    ---
    words: 单词列表
    idx: input word的索引号
    window_size: 窗口大小
    '''
    target_window = np.random.randint(1, window_size + 1)
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    targets = set(words[start_point:idx] + words[idx + 1:end_point + 1])
    return list(targets)


def get_batches(words, batch_size, window_size=5):
    n_batches = len(words) // batch_size
    words = words[:n_batches * batch_size]
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx + batch_size]
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_targets(words, i, window_size)
            x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)
        yield x, y


train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')

vocab_size = len(id_to_word)
embedding_dim = 200

with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs)

n_sampled = 100

with train_graph.as_default():
    softmax_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_dim], stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(vocab_size))

    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

with train_graph.as_default():
    valid_size = 7
    valid_window = 100
    valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
    valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size // 2))

    valid_examples = [
        word_to_id['詹姆斯'],
        word_to_id['詹姆斯'],
        word_to_id['詹姆斯'],
        word_to_id['詹姆斯'],
        word_to_id['詹姆斯'],
        word_to_id['詹姆斯'],
        word_to_id['詹姆斯']
    ]
    valid_size = len(valid_examples)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm

    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

epochs = 10
batch_size = 1000
window_size = 10

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()

        for x, y in batches:
            feed = {inputs: x, labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / 100),
                      "{:.4f} sec/batch".format((end - start) / 100))
                loss = 0
                start = time.time()

            if iteration % 1000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = id_to_word[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                    log = 'Nearest to [%s]:' % valid_word
                    for k in range(top_k):
                        close_word = id_to_word[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
            iteration += 1

    save_path = saver.save(sess, 'checkpoints/word2vec.ckpt')
    embed_mat = sess.run(normalized_embedding)
