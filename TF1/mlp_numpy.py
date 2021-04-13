# !usr/bin/env python
# -*- coding:utf-8 _*-
# author:chenmeng
# datetime:2021/3/18 10:55

import numpy as np

B, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(B, D_in)  # [64, 1000]
y = np.random.randn(B, D_out)  # [64, 10]

w1 = np.random.randn(D_in, H)  # [1000, 100]
w2 = np.random.randn(H, D_out)  # [100, 10]

learning_rate = 1e-6

for t in range(500):
    # 前向传播，包含输入层，激活函数relu，和输出层
    h = x.dot(w1)  # [64, 100]
    # 隐层，使用Relu激活函数
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)  # [64,10]

    # loss使用均方误差
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # 计算w1和w2的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    # 因为relu的导数在x>0时为1，所以这里导数与对整个relu隐层的导数一样
    grad_h = grad_h_relu.copy()
    # 在x<0时，导数为0
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # 更新参数
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
