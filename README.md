# nlp_basic_model

## Introduction
实现NLP中常用的基础模型，计划先用使用TensorFlow 1.x实现，然后使用TensorFlow 2.x 和 pytorch复现一遍。

## Motivation
感觉自己太菜了，每次要用一些基础模型的时候都去github去git clone，现在想利用业余时间通过比较主流的框架把这些基础模型实现一遍。

## Requirements
> python>3.6  
> TensorFlow > 1.12  
## Dataset
分类任务使用数据集为[THUCNnews](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews)，
为了快速训练，使用data_helper.py将数据集进行切分，只使用一小部分数据进行训练。

## Changelog 
- 单层LSTM（LSTM/lstm_single.py）

实现单向LSTM用作分类任务。其中添加了TensorBoard，并且实现的参数相对比较灵活，可以进行调整，data_loader可以进行复用。

- 双向LSTM(LSTM/bi_lstm.py)

实现双向LSTM用作分类任务。与单向LSTM共用data_loader。

- 多层双向LSTM

对于多层双向LSTM具体是什么样子的，我一直认为是多个双向LSTM层层堆积出来，而不是多层正向LSTM然后在加上多层反向LSTM堆积出来的。然后现在在众多“教程”中，实现出来的都是后者，并且没有一个清晰的解释。<br>

![fake multi layer bi-lstm](https://i.loli.net/2020/06/08/ZjsNFpgI3aioWQ6.png)<br>
图一 fake multi layer bi-lstm<br>
![ture multi layer bi-lstm](https://i.loli.net/2020/06/08/QfZnMFJiryTY8Ap.png)<br>
图二 true multi layer bi-lstm<br>
在众多“教程”中，实现的方法一般都是使用tf.nn.bidirectional_dynamic_rnn，在这之前将正向和反向LSTM进行堆叠，然后放入到双向LSTM中。直到我看到https://blog.csdn.net/u012436149/article/details/71080601
这篇文章，看到这个博主自己实现的堆叠多层双向LSTM直呼高手，并一直用这个博主实现的API，后来发现TensorFlow官方自己实现了这个API，tf.contrib.rnn.stack_bidirectional_dynamic_rnn，API的介绍在http://tensorflow.biotecan.com/python/Python_1.8/tensorflow.google.cn/api_docs/python/tf/contrib/rnn/stack_bidirectional_dynamic_rnn.html，
具体可以看https://stackoverflow.com/questions/49242266/difference-between-bidirectional-dynamic-rnn-and-stack-bidirectional-dynamic-rnn的讨论，上边的图也是来自这个讨论帖，说的很清楚。