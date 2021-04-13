# !usr/bin/env python
# -*- coding:utf-8 _*-
# author:chenmeng
# datetime:2020/8/31 13:46

import tensorflow as tf

from bert import modeling


class Model(object):
    def __init__(self, bert_config, label_size, learning_rate):
        self.inputs_seq = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_seq')
        self.inputs_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
        self.inputs_segment = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_segment')
        self.label = tf.placeholder(dtype=tf.int32, shape=[None, label_size], name='label')

        bert_model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=self.inputs_seq,
            input_mask=self.inputs_mask,
            token_type_ids=self.inputs_segment,
            use_one_hot_embeddings=False)

        # bert_model.get_sequence_output()获取的是BERT的最后一层，维度是：[batch_size, seq_length, hidden_size]，
        # bert_model.get_pooled_output()将get_sequence_output()的维度转换成[batch_size, hidden_size]
        bert_outputs = bert_model.get_pooled_output()

        with tf.variable_scope('loss'):
            logits = tf.layers.dense(bert_outputs, label_size)
            self.probabilities = tf.sigmoid(logits)
            labels = tf.cast(self.label, tf.float32)
            pre_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
            self.loss = tf.reduce_mean(pre_example_loss)

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
