# !usr/bin/env python
# -*- coding:utf-8 _*-
# author:chenmeng
# datetime:2020/8/11 10:52

import tensorflow as tf
from bert import modeling


class ModelBertCRF(object):
    def __init__(self,
                 bert_config,
                 vocab_size_bio,
                 use_lstm,
                 use_crf,
                 num_units,
                 max_seq_len,
                 learning_rate_bert,
                 learning_rate_crf):
        self.inputs_seq = tf.placeholder(dtype=tf.int32, shape=[None, None], name='inputs_seq') # batch_size * (sequence_len + 2)
        self.inputs_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name='inputs_mask') # batch_size * (sequence_len + 2)
        self.inputs_segment = tf.placeholder(dtype=tf.int32, shape=[None, None], name='inputs_segment') # batch_size * (sequence_len + 2)
        self.outputs_seq = tf.placeholder(dtype=tf.int32, shape=[None, None], name='outputs_seq') # batch_size * (sequence_len + 2)

        inputs_seq_len = tf.reduce_sum(self.inputs_mask, axis=-1) # batch_size

        bert_model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=self.inputs_seq,
            input_mask=self.inputs_mask,
            token_type_ids=self.inputs_segment,
            use_one_hot_embeddings=False
        )

        bert_outputs = bert_model.get_sequence_output() # batch_size * (sequence_len + 2) * dim

        if not use_lstm:
            hiddens = bert_outputs
        else:
            with tf.variable_scope('bilstm'):
                cell_fw = tf.nn.rnn_cell.LSTMCell(num_units)
                cell_bw = tf.nn.rnn_cell.LSTMCell(num_units)
                ((rnn_fw_outputs, rnn_bw_outputs),
                 (rnn_fw_final_state, rnn_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=bert_outputs,
                    sequence_length=inputs_seq_len,
                    dtype=tf.float32
                )
                rnn_outputs = tf.add(rnn_fw_outputs, rnn_bw_outputs) # batch_size * (sequence_len + 2) * dim
            hiddens = rnn_outputs

        with tf.variable_scope('projection'):
            logits_seq = tf.layers.dense(hiddens, vocab_size_bio) # batch_size * (sequence_len + 2) * vocab_size
            probs_seq = tf.nn.softmax(logits_seq)

            if not use_crf:
                preds_seq = tf.argmax(probs_seq, axis=-1, name='preds_seq') # batch_size * vocab_size
            else:
                log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_seq, self.outputs_seq,
                                                                                      inputs_seq_len)
                preds_seq, crf_scores = tf.contrib.crf.crf_decode(logits_seq, transition_matrix, inputs_seq_len)

        self.outputs = preds_seq

        with tf.variable_scope('loss'):
            if not use_crf:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_seq, labels=self.outputs_seq) # batch_size * (sequence_len + 2)
                # tf.sequence_mask(),默认最大长度为输入的tensor中的最大长度，因为我们自己设定了最大长度，这里需要传一个参数，并且加上了[SEQ][CLS]所以需要+2
                masks = tf.sequence_mask(inputs_seq_len, maxlen=max_seq_len + 2, dtype=tf.float32) # batch_size * (sequence_len + 2)
                loss = tf.reduce_sum(loss * masks, axis=-1) / tf.cast(inputs_seq_len, tf.float32) # batch_size
            else:
                loss = -log_likelihood / tf.cast(inputs_seq_len, tf.float32)

        self.loss = tf.reduce_mean(loss)

        '''
        其实将不同的参数使用不同的学习率进行训练就是将optimizer.minimize(loss, varlist)拆分开来写。
        先来看一下minimize的源码：先进行compute_gradients()，然后进行apply_gradients().
        也就是先计算梯度，然后使用梯度来更新参数。
        那么应用到不同的参数使用不同的学习也是一样的流程：先取到不同的参数集合，然后使用loss和不同的参数集合
        计算不同的参数的梯度，最后使用不同的梯度来更新不同的参数。其中为了防止梯度爆炸和梯度消失对梯度进行了
        剪切，让参数的更新限制在一个合适的范围内。
        https://cloud.tencent.com/developer/article/1375874
        '''
        with tf.variable_scope('opt'):
            params_of_bert = []
            params_of_others = []
            for var in tf.trainable_variables():
                vname = var.name
                if vname.startswith('bert'):
                    params_of_bert.append(var)
                else:
                    params_of_others.append(var)
            opt_bert = tf.train.AdamOptimizer(learning_rate_bert)
            opt_others = tf.train.AdamOptimizer(learning_rate_crf)
            # 计算梯度
            gradients_bert = tf.gradients(self.loss, params_of_bert)
            gradients_others = tf.gradients(self.loss, params_of_others)
            # 为了防止梯度爆炸或者梯度消失，对梯度进行剪切。让参数的更新限制在一个合适的范围。
            # https://blog.csdn.net/u013713117/article/details/56281715
            gradients_bert_clipped, norm_bert = tf.clip_by_global_norm(gradients_bert, 5.0)
            gradients_others_clipped, norm_others = tf.clip_by_global_norm(gradients_others, 5.0)
            # 更新参数
            train_op_bert = opt_bert.apply_gradients(zip(gradients_bert_clipped, params_of_bert))
            train_op_others = opt_others.apply_gradients(zip(gradients_others_clipped, params_of_others))

        self.train_op = (train_op_bert, train_op_others)
