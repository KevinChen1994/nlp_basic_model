# !usr/bin/env python
# -*- coding:utf-8 _*-
# author:chenmeng
# datetime:2020/6/11 18:16

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
save_dir = 'checkpoints/multi_kernel_size_textcnn'
save_path = os.path.join(save_dir, 'multi_kernel_size_textcnn')
tensorboard_dir = 'tensorboard/multi_kernel_size_textcnn'


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class CNNConfig(object):
    num_classes = 10
    seq_length = 128
    num_filters = 256  # 卷积核个数
    kernel_size = [3, 4, 5, 6]  # 卷积核大小
    vocab_min_size = 2
    vocab_size = 5000
    embedding_dim = 100
    dropout_prob = 0.5
    learning_rate = 1e-3
    decay_rate = 0.1  # 学习率衰减比率
    decay_steps = 2000  # 衰减步数
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
            # 需要将输入转换成四维的矩阵，最后一维为深度
            # embeddings_inputs_expanded[batch_size, sequence_length, embedding_dim, input_depth]
            self.embeddings_inputs_expanded = tf.expand_dims(embeddings_inputs, -1)

        with tf.name_scope('cnn_layer'):
            pools = []
            for i, kernel_size in enumerate(self.config.kernel_size):
                # shape = [kernel_height, kernel_width, input_depth, output_depth]
                filter_weights = tf.get_variable(name='weights' + str(i),
                                                 shape=[kernel_size, self.config.embedding_dim, 1,
                                                        self.config.num_filters],
                                                 initializer=tf.variance_scaling_initializer(), dtype=tf.float32)
                biases = tf.get_variable(name='biases' + str(i), shape=[self.config.num_filters],
                                         initializer=tf.zeros_initializer(),
                                         dtype=tf.float32)
                ''''
                strides=[batch_stride, height_stride, width_stride, depth_stride]，第一个和第四个维度要求为1，因为卷积层的步长只对矩阵的长和宽有效。
                padding  VALID: 不填充;  SAME: 全0填充;
                不同的填充方法最终导致的输出矩阵的大小是不一样的，具体的计算方法为：
                使用全0填充 output_height=[input_height/stride_height]  output_width=[input_width/stride_width]
                不填充 output_height=[(input_height-filter_height+1)/stride_height]  output_width=[(input_width-filter_width+1)/stride_width]
                '''
                conv = tf.nn.conv2d(self.embeddings_inputs_expanded, filter_weights, strides=[1, 1, 1, 1],
                                    padding='VALID')
                conv = tf.nn.relu(tf.nn.bias_add(conv, biases), name='conv')

                # ksize=[batch_kernel_size, height_kernel, width_kernel, depth_kernel_size]，第一个和第四个维度通常为1。
                # 这里设置过滤器的尺寸为在卷积完之后的高度和宽度，目的是池化完之后的结果第二个维度和第三个维度都为1，方便后边计算。
                pool = tf.nn.max_pool(conv, ksize=[1, self.config.seq_length - kernel_size + 1, 1, 1],
                                      strides=[1, 1, 1, 1], padding='VALID', name='pool')
                pools.append(pool)

            pools = tf.concat(pools, 3)
            # 将池化的结果修改维度，修改成二维矩阵，因为第二维和第三维本身就是1，那么最后的结果为[batch_size,num_filters*len(kernel_size)]这里也就是pools最后一个维度的值
            h = tf.reshape(pools, [-1, pools.get_shape()[3].value])
            w = tf.get_variable(name='w', shape=[pools.get_shape()[3].value, self.config.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b = tf.get_variable(name='b', shape=[self.config.num_classes],
                                initializer=tf.zeros_initializer(), dtype=tf.float32)
            fc = tf.matmul(h, w) + b
            self.logits = tf.nn.dropout(fc, self.config.dropout_prob)

        with tf.name_scope('optimize_layer'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.label))

            # 动态学习率，随着训练步数进行衰减
            global_step = tf.Variable(0)
            self.dynamic_learning_rate = tf.train.exponential_decay(learning_rate=self.config.learning_rate,
                                                                    global_step=global_step,
                                                                    decay_rate=self.config.decay_rate,
                                                                    decay_steps=self.config.decay_steps)

            self.optimizer = tf.train.AdamOptimizer(self.dynamic_learning_rate).minimize(self.loss)

        with tf.name_scope('score'):
            self.predict_label = tf.argmax(self.logits, 1)
            correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.label, 1))
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
