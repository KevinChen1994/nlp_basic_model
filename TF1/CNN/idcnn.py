# -*- coding:utf-8 -*-
# author: chenmeng
# datetime:2022/2/9 17:47
# Description:

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
save_dir = 'checkpoints/idcnn'
save_path = os.path.join(save_dir, 'idcnn')
tensorboard_dir = 'tensorboard/idcnn'


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class CNNConfig(object):
    num_classes = 10
    seq_length = 128
    num_filters = 256  # 卷积核个数
    kernel_size = 3  # 卷积核大小
    vocab_min_size = 2
    vocab_size = 5000
    embedding_dim = 100
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
        self.sequence_lengths = tf.placeholder(tf.int32, [None], name='sequence_lengths')

        self.cnn()

    def random_embedding(self, vocab_size, embedding_dim):
        embedding_mat = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
        embedding_mat = np.float32(embedding_mat)
        return embedding_mat

    def cnn(self):
        with tf.name_scope('embedding_layer'):
            embeddings = self.random_embedding(self.config.vocab_size, self.config.embedding_dim)
            embedding = tf.Variable(embeddings, dtype=tf.float32, trainable=True, name='word_embedding')
            embeddings_inputs = tf.nn.embedding_lookup(embedding, self.content)
            # embeddings_inputs[batch_size, sequence_length, embedding_dim]
            # 需要将输入转换成四维的矩阵，在第二维进行扩充维度
            # embeddings_inputs_expanded[batch_size, 1, sequence_length, embedding_dim]
            self.embeddings_inputs_expanded = tf.expand_dims(embeddings_inputs, 1)

        with tf.name_scope('idcnn_layer'):
            '''
            先使用一层CNN得到输出作为IDCNN的输入，因为IDCNN是循环进行卷积的，所以要求in_channel和out_channel一样，
            不然在循环的过程中，需要不断去调整卷积核的尺寸。
            '''
            # shape = [kernel_height, kernel_width, input_depth, output_depth]
            filter_weights = tf.get_variable(name='weights',
                                             shape=[1, self.config.kernel_size, self.config.embedding_dim,
                                                    self.config.num_filters],
                                             initializer=tf.variance_scaling_initializer(), dtype=tf.float32)
            biases = tf.get_variable(name='biases', shape=[self.config.num_filters], initializer=tf.zeros_initializer(),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(self.embeddings_inputs_expanded, filter_weights, strides=[1, 1, 1, 1],
                                padding='SAME')  # [batch_size, 1, sequence_length, num_filter]
            layerInput = tf.nn.relu(tf.nn.bias_add(conv, biases), name='conv')
            layers = [
                {
                    'dilation': 1
                },
                {
                    'dilation': 1
                },
                {
                    'dilation': 2
                },
            ]
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            for j in range(4):
                for i in range(len(layers)):
                    dilation = layers[i]['dilation']
                    isLast = True if i == (len(layers) - 1) else False
                    with tf.variable_scope('atrous-conv-layer-%d' % i, reuse=tf.AUTO_REUSE):
                        w = tf.get_variable("filterW", shape=[1, self.config.kernel_size, self.config.num_filters,
                                                              self.config.num_filters],
                                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.config.num_filters])
                        conv = tf.nn.atrous_conv2d(layerInput, w, rate=dilation, padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.config.num_filters
                        layerInput = conv
            idcnn_outputs = tf.concat(finalOutFromLayers,
                                      axis=3)  # [batch_size, 1, seq_len, num_filter*4]
            idcnn_outputs = tf.nn.relu(idcnn_outputs)
            idcnn_outputs = tf.squeeze(idcnn_outputs, [1])
            idcnn_outputs = tf.reshape(idcnn_outputs, (-1, self.config.seq_length * totalWidthForLastDim))

            fc = tf.layers.dense(idcnn_outputs, self.config.num_classes)
            self.logtis = tf.nn.dropout(fc, self.config.dropout_prob)

        with tf.name_scope('optimize_layer'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logtis, labels=self.label))

            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('score'):
            self.predict_label = tf.argmax(self.logtis, 1)
            correct_pred = tf.equal(tf.argmax(self.logtis, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def evaluate(sess, x, y, seq_lens, batch_size):
    data_len = len(x)
    batch_eval = batch_iter(x, y, seq_lens, batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch, seq_lens_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {model.content: x_batch, model.label: y_batch, model.sequence_lengths: seq_lens_batch}
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
    x_train, y_train, seq_lens_train = process_file(train_dir, word_to_id, label_to_id, config.seq_length)
    x_val, y_val, seq_lens_val = process_file(val_dir, word_to_id, label_to_id, config.seq_length)
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
        batch_train = batch_iter(x_train, y_train, seq_lens_train, config.batch_size)
        for x_batch, y_batch, seq_lens_batch in batch_train:
            feed_dict = {model.content: x_batch, model.label: y_batch, model.sequence_lengths: seq_lens_batch}

            # 将训练结果写如到TensorBoard中
            if total_batch % config.save_pre_batch == 0:
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            # 输出训练集和验证集的结果，并保存最好的模型
            if total_batch % config.print_pre_batch == 0:
                loss_train, acc_train = session.run([model.loss, model.accuracy], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val, seq_lens_val, config.batch_size)
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
    x_test, y_test, seq_lens_test = process_file(test_dir, word_to_id, label_to_id, config.seq_length)
    content_test, label_test = read_corpus(test_dir)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # 读取模型
    saver.restore(sess=session, save_path=save_path)

    print('Start testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test, seq_lens_test, config.batch_size)
    msg = 'Test Loss: {0:>2.2f}, Test Acc: {1:>2.2%}'
    print(msg.format(loss_test, acc_test))

    data_len = len(x_test)
    num_batch = int((data_len - 1) / config.batch_size) + 1
    predict_result = np.zeros(shape=[len(x_test)], dtype=np.int32)
    for i in range(num_batch):
        start = i * config.batch_size
        end = min(data_len, start + config.batch_size)
        feed_dict = {model.content: x_test[start:end], model.label: y_test[start:end],
                     model.sequence_lengths: seq_lens_test[start:end]}
        predict_result[start:end] = session.run(model.predict_label, feed_dict=feed_dict)

    print('Writing predict result to predict.txt...')
    with open('predict.txt', 'w', encoding='utf-8') as f:
        for i in range(len(predict_result)):
            f.write(id_to_label[predict_result[i]] + '\t' + content_test[i] + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python lstm_single.py [train / test]""")
    config = CNNConfig()
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
