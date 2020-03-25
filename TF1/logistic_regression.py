# -*- coding:utf-8 -*-
# author:chenmeng
# datetime:2020/3/25 15:37
# software: PyCharm

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.01
epoch = 100
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

# 每张image的shape是[28,28]，合并到一行以后就是[,748]
trainX = mnist.train.images
# label的shape是[,10]
trainY = mnist.train.labels
testX = mnist.test.images
testY = mnist.test.labels

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])
# stddev 正态分布的标准差，默认是1.0
w = tf.Variable(tf.random_normal([784, 10], stddev=0.01))


def model(x, w):
    return tf.matmul(x, w)


x_model = model(x, w)
# tf.nn.softmax_cross_entropy_with_logits()返回值是一个tensor，并不是一个值，所以需要做一个取平均的操作
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x_model))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
predict_op = tf.argmax(x_model, axis=1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epoch):
        for start, end in zip(range(0, len(trainX), 128), range(128, len(trainX) + 1, 128)):
            _, loss_ = sess.run([train_op, loss], feed_dict={x: trainX[start:end], y: trainY[start:end]})
        print('epoch', i + 1, 'loss', loss_)
        # 预测准确率
        print(np.mean(np.argmax(testY, axis=1) == sess.run(predict_op, feed_dict={x: testX})))
