# nlp_basic_model

## Introduction
实现NLP中常用的基础模型，计划先用使用TensorFlow 1.x实现，然后使用TensorFlow 2.x 和 pytorch复现一遍。

## Motivation
感觉自己太菜了，每次要用一些基础模型的时候都去github去git clone，现在想利用业余时间通过比较主流的框架把这些基础模型实现一遍。

## Requirements
> python>3.6  
> TensorFlow > 1.12  
## dataset
分类任务使用数据集为[THUCNnews](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews)，
为了快速训练，使用data_helper.py将数据集进行切分，只使用一小部分数据进行训练。

## Changelog 
- 单层LSTM（LSTM/lstm_single.py）

实现单向LSTM用作分类任务。其中添加了TensorBoard，并且实现的参数相对比较灵活，可以进行调整，data_loader可以进行复用。

- 双向LSTM(LSTM/bi_lstm.py)

实现双向LSTM用作分类任务。与单向LSTM共用data_loader。


