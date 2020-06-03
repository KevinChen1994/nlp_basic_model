# -*- coding:utf-8 -*-
# author:chenmeng
# datetime:2020/3/28 22:22
# software: PyCharm

import tensorflow as tf
import numpy as np
import sys
import os
import time
from datetime import timedelta
from TF1.LSTM.data_loader import read_corpus, read_vocab, read_labels, vocab_build, process_file, batch_iter

base_dir = '../../data/small_cnews'
train_dir = os.path.join(base_dir, 'train.txt')
test_dir = os.path.join(base_dir, 'test.txt')
val_dir = os.path.join(base_dir, 'val.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')
label_dir = os.path.join(base_dir, 'label.txt')
save_dir = 'checkpoints'
save_path = os.path.join(save_dir, 'lstm_single')
tensorboard_dir = 'tensorboard/'


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class LSTMConfig(object):
    num_classes = 10
    seq_length = 128
    vocab_min_size = 2
    vocab_size = 5000
    embedding_dim = 100
    num_units = 128
    dropout_prob = 0.5
    learning_rate = 1e-3
    batch_size = 128
    epoch = 20

    print_pre_batch = 100  # 每多少轮输出一次结果
    save_pre_batch = 10  # 每多少轮存入到TensorBoard


class model(object):
    def __init__(self, config):
        self.config = config

        self.label = tf.placeholder(tf.int32, [None, config.num_classes], name='label')
        self.content = tf.placeholder(tf.int32, [None, config.seq_length], name='content')

        self.lstm()

    def random_embedding(self, vocab_size, embedding_dim):
        embedding_mat = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
        embedding_mat = np.float32(embedding_mat)
        return embedding_mat

    def lstm(self):
        with tf.name_scope('embedding_layer'):
            embeddings = self.random_embedding(self.config.vocab_size, self.config.embedding_dim)
            embedding = tf.Variable(embeddings, dtype=tf.float32, trainable=True, name='word_embedding')
            self.embeddings_inputs = tf.nn.embedding_lookup(embedding, self.content)

        with tf.name_scope('lstm_layer'):
            cell = tf.nn.rnn_cell.LSTMCell(self.config.num_units)
            outputs, states = tf.nn.dynamic_rnn(cell, self.embeddings_inputs, dtype=tf.float32)
            w = tf.get_variable(name='w', shape=[self.config.num_units, self.config.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b = tf.get_variable(name='b', shape=[self.config.num_classes],
                                initializer=tf.zeros_initializer(), dtype=tf.float32)
            # outputs[batch_size, num_steps, num_units] -> [num_steps, batch_size, num_units]
            outputs = tf.transpose(outputs, [1, 0, 2])
            outputs = outputs[-1]
            fc = tf.matmul(outputs, w) + b
            self.logtis = tf.nn.dropout(fc, self.config.dropout_prob)

        with tf.name_scope('optimize_layer'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logtis, labels=self.label))

            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('score'):
            self.predict_label = tf.argmax(self.logtis, 1)
            correct_pred = tf.equal(tf.argmax(self.logtis, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def evaluate(sess, x, y, batch_size):
    data_len = len(x)
    batch_eval = batch_iter(x, y, batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {model.content: x_batch, model.label: y_batch}
        loss, acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss / data_len, total_acc / data_len


def train():
    # 配置TensorBoard和Saver
    print('Configuring TensorBoard and Saver...')
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 处理数据
    print('Loading training data and validation data...')
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, label_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, label_to_id, config.seq_length)
    print('Time usage:', get_time_dif(start_time))

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # 将图添加到TensorBoard中
    writer.add_graph(session.graph)

    # 开始训练
    print('Start training...')
    start_time = time.time()
    best_acc_val = 0.0
    total_batch = 0

    for epoch in range(config.epoch):
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = {model.content: x_batch, model.label: y_batch}

            # 将训练结果写如到TensorBoard中
            if total_batch % config.save_pre_batch == 0:
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            # 输出训练集和验证集的结果，并保存最好的模型
            if total_batch % config.print_pre_batch == 0:
                loss_train, acc_train = session.run([model.loss, model.accuracy], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val, config.batch_size)
                # 每次只保存最好的模型
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    saver.save(session, save_path)
                    improved_str = '*'
                else:
                    improved_str = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>2}, Train Loss: {1:>2.2f}, Train Acc: {2:>2.2%}, ' \
                      'Val Loss: {3:>2.2f}, Val Acc: {4:>2.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optimizer, feed_dict=feed_dict)
            total_batch += 1


def test():
    print('Loading test data...')
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, label_to_id, config.seq_length)
    content_test, label_test = read_corpus(test_dir)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # 读取模型
    saver.restore(sess=session, save_path=save_path)

    print('Start testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test, config.batch_size)
    msg = 'Test Loss: {0:>2.2f}, Test Acc: {1:>2.2%}'
    print(msg.format(loss_test, acc_test))

    data_len = len(x_test)
    num_batch = int((data_len - 1) / config.batch_size) + 1
    predict_result = np.zeros(shape=[len(x_test)], dtype=np.int32)
    for i in range(num_batch):
        start = i * config.batch_size
        end = min(data_len, start + config.batch_size)
        feed_dict = {model.content: x_test[start:end], model.label: y_test[start:end]}
        predict_result[start:end] = session.run(model.predict_label, feed_dict=feed_dict)

    print('Writing predict result to predict.txt...')
    with open('predict.txt', 'w', encoding='utf-8') as f:
        for i in range(len(predict_result)):
            f.write(id_to_label[predict_result[i]] + '\t' + content_test[i] + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python lstm_single.py [train / test]""")
    config = LSTMConfig()
    if not os.path.exists(vocab_dir):
        vocab_build(train_dir, vocab_dir, config.vocab_min_size)

    labels, label_to_id, id_to_label = read_labels(label_dir)
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = model(config)
    if sys.argv[1] == 'train':
        train()
    else:
        test()
