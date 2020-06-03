# nlp_basic_model

## Introduction
实现NLP中常用的基础模型，计划先用使用TensorFlow 1.x实现，然后使用TensorFlow 2.x 和 pytorch复现一遍。

## Motivation
感觉自己太菜了，每次要用一些基础模型的时候都去github去git clone，现在想利用业余时间通过比较主流的框架把这些基础模型实现一遍。

## Requirements
> python>3.6  
> TensorFlow > 1.12  


## Changelog 
- lstmsingle

实现单向LSTM用作分类任务。其中添加了TensorBoard，并且实现的参数相对比较灵活，可以进行调整，data_loader可以进行复用。
