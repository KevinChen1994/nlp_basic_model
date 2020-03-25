# -*- coding:utf-8 -*-
# author:chenmeng
# datetime:2020/3/25 14:46
# software: PyCharm

import tensorflow as tf
import numpy as np

learning_rate = 0.01
epoch = 100

trainX = np.linspace(-1, 1, 101)
# np.random.rand(*dn) 返回的是dn形状的[0,1)中的随机样本
# *train_X.shape中 * 的意思表示将一个数组解压，比如a = [1,  2,  3]，那*a则是  1,  2,  3，把数组打开了，这么用是因为randn函数的参数需求
# trainY是两倍的trainX在加上一些随机噪声
trainY = 2 * trainX + np.random.rand(*trainX.shape) * 0.33

x = tf.placeholder('float', name='x')
y = tf.placeholder('float', name='y')
# 权重
w = tf.Variable(0.0, name='weights')


# 模型就是x*w
def model(x, w):
    return tf.multiply(x, w)


# 模型的输出值
y_model = model(x, w)
# 损失用差的平方
loss = tf.square(y - y_model)
# 使用梯度下降进行训练模型
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epoch):
        for (train_x, train_y) in zip(trainX, trainY):
            # 使用feed_dict进行给模型输入数据，将两个placeholder按要求输入就可以
            _, loss_ = sess.run([train_op, loss], feed_dict={x: train_x, y: train_y})
        # 输出每一个epoch的损失
        print('epoch:', i + 1, 'loss:', loss_)
    # 最终输出训练出来的权重w，这里应该是一个接近2的数
    print(sess.run(w))
    # 做一个预测，输入x为10，y随便写一个，因为模型的输入不需要y，这里需要预测的是y，输出应该是一个接近20的数
    print(sess.run(y_model, feed_dict={x: 10, y: 0}))
