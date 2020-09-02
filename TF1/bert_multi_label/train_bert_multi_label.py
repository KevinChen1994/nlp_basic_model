# !usr/bin/env python
# -*- coding:utf-8 _*-
# author:chenmeng
# datetime:2020/8/31 15:05
import argparse
import tensorflow as tf
import numpy as np
import os
import logging
import time
from tqdm import tqdm

from bert import modeling
from TF1.bert_multi_label.model_bert_multi_label import Model
from TF1.bert_multi_label.data_helper import load_vocabulary
from TF1.bert_multi_label.data_helper import read_labels
from TF1.bert_multi_label.data_helper import data_process
from TF1.bert_multi_label.data_helper import batch_iter

## hyperparameters
parser = argparse.ArgumentParser(description='bert-multi-label')

parser.add_argument('--bert_vocab_path', type=str, default='../../bert_ckpt/vocab.txt',
                    help='bert vocab path')
parser.add_argument('--bert_config_path', type=str, default='../../bert_ckpt/bert_config.json',
                    help='bert config path')
parser.add_argument('--bert_ckpt_path', type=str, default='../../bert_ckpt/bert_model.ckpt',
                    help='bert checkpoint path')
parser.add_argument('--data_path', type=str, default='../../data/amazon',
                    help='data path, this path includes train data test data dev data')
parser.add_argument('--output_path', type=str, default='./output', help='output path')
parser.add_argument('--max_sequence_len', type=int, default=32, help='max length of sequence')
parser.add_argument('--batch_size', type=int, default=500, help='sample of each minibatch')
parser.add_argument('--epoch', type=int, default=10, help='num of training epoch')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
parser.add_argument('--mode', type=str, default='train', help='train/test')
parser.add_argument('--model_name', type=str, default='1597994676', help='mode name for test')
args = parser.parse_args()

# 如果指定的GPU不存在，TF自动分配
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True

timestamp = str(int(time.time())) if args.mode == 'train' else args.model_name
output_path = os.path.join(args.output_path, timestamp)
if not os.path.exists(output_path):
    os.makedirs(output_path)

ckpt_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
tensorboard_path = os.path.join(output_path, 'tensorboard')
if not os.path.exists(tensorboard_path):
    os.mkdir(tensorboard_path)
result_path = os.path.join(output_path, "results")
if not os.path.exists(result_path):
    os.makedirs(result_path)
log_path = os.path.join(result_path, "run_train.log") if args.mode == 'train' else os.path.join(result_path,
                                                                                                "run_test.log")
if os.path.exists(log_path):
    os.remove(log_path)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
fhlr = logging.FileHandler(log_path)
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)

logger.info(str(args))

logger.info('loading vocab...')
w2i_char, i2w_char = load_vocabulary(args.bert_vocab_path)
w2i_label, i2w_label = read_labels(os.path.join(args.data_path, 'categories.csv'))


def train():
    logger.info('loading data...')
    seq_id_train, label_train, input_mask_train, input_segment_train = data_process(
        os.path.join(args.data_path, 'train.txt'), w2i_char, w2i_label, args.max_sequence_len)
    seq_id_val, label_val, input_mask_val, input_segment_val = data_process(
        os.path.join(args.data_path, 'dev.txt'), w2i_char, w2i_label, args.max_sequence_len)

    logger.info('building model...')
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_path)
    logger.info(bert_config.to_json_string())

    model = Model(bert_config=bert_config,
                  label_size=len(w2i_label),
                  learning_rate=args.learning_rate)
    # 加载可以训练参数
    tvars = tf.trainable_variables()
    # 加载BERT模型
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                               args.bert_ckpt_path)

    tf.train.init_from_checkpoint(args.bert_ckpt_path, assignment_map)

    logger.info('configuring tensorboard...')
    tf.summary.scalar('loss', model.loss)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_path)

    logger.info('start training...')

    with tf.Session(config=tf_config) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3)
        writer.add_graph(session.graph)

        losses = []
        batches = 0
        for epoch in range(args.epoch):
            batch_train = batch_iter(seq_id_train, label_train, input_mask_train, input_segment_train,
                                     args.batch_size)
            time_epoch_start = time.time()
            for inputs_seq_batch, label_batch, inputs_mask_batch, inputs_segment_batch in batch_train:
                if batches == 0:
                    logger.info("###### shape of a batch #######")
                    logger.info("inputs_seq: " + str(inputs_seq_batch.shape))
                    logger.info("inputs_mask: " + str(inputs_mask_batch.shape))
                    logger.info("inputs_segment: " + str(inputs_segment_batch.shape))
                    logger.info("outputs_seq: " + str(label_batch.shape))
                    logger.info("###### preview a sample #######")
                    logger.info("input_seq:" + " ".join([i2w_char[i] for i in inputs_seq_batch[0]]))
                    logger.info("input_mask :" + " ".join([str(i) for i in inputs_mask_batch[0]]))
                    logger.info("input_segment :" + " ".join([str(i) for i in inputs_segment_batch[0]]))
                    logger.info("output_seq: " + " ".join(
                        [i2w_label[index] for index, i in enumerate(label_batch[0]) if i != 0]))
                    logger.info("###############################")
                feed_dict = {model.inputs_seq: inputs_seq_batch, model.label: label_batch,
                             model.inputs_mask: inputs_mask_batch, model.inputs_segment: inputs_segment_batch}
                loss, _ = session.run([model.loss, model.optimizer], feed_dict)
                losses.append(loss)

                if batches % 100 == 0:
                    logger.info('epoch: {}'.format(epoch))
                    logger.info('batch: {}'.format(batches))
                    logger.info('loss: {}'.format(sum(losses) / len(losses)))
                    logger.ingo('time: {{'.format(time.time() - time_epoch_start))
                    time_epoch_start = time.time()
                    losses = []

                    s = session.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, batches)

                    # 在验证集进行验证
                    valid(model, session, seq_id_val, label_val, input_mask_val, input_segment_val)
                    ckpt_save_path = os.path.join(ckpt_path, 'model.ckpt')
                    logger.info('path of ckpt: {}-{}'.format(ckpt_save_path, batches))
                    saver.save(session, ckpt_save_path, global_step=batches)
                batches += 1


'''
这里用到的计算方式是宏平均F1，计算方式是按行计算，不管一共有多少个标签，计算的是总的评价，先计算出全部的precision和recall然后计算F1值，
如果想要计算每一个类别的acc precision recall f1也是可以的，只要在计算的时候按照列来算就可以了，这样每一列就是一个标签。
'''


def valid(model, sess, seq_id_val, label_val, input_mask_val, input_segment_val):
    time_valid = time.time()
    predict_label = []
    real_label = []
    p_sum = 0
    r_sum = 0
    hits = 0
    batch_train = batch_iter(seq_id_val, label_val, input_mask_val, input_segment_val, args.batch_size)
    for i, (inputs_seq_batch, label_batch, input_mask_batch, input_segment_batch) in enumerate(batch_train):
        feed_dict = {model.inputs_seq: inputs_seq_batch, model.label: label_batch,
                     model.inputs_mask: input_mask_batch, model.inputs_segment: input_segment_batch}
        predict_label_prob_batch = sess.run(model.probabilities, feed_dict=feed_dict)
        for i in range(len(predict_label_prob_batch)):
            predict_label.append([i for i, prob in enumerate(predict_label_prob_batch[i]) if prob > 0.5])
            real_label.append([i for i, prob in enumerate(label_batch[i]) if prob == 1])

        for predict_label_, real_label_ in zip(predict_label, real_label):
            p_sum += len(predict_label_)
            r_sum += len(real_label_)
            for label in predict_label_:
                if label in real_label_:
                    hits += 1
    p = hits * 100 / p_sum if p_sum != 0 else 0
    r = hits * 100 / r_sum if r_sum != 0 else 0
    f1 = (2 * p * r) / (p + r) if p + r > 0 else 0
    logger.info('Precision:{}%'.format(p))
    logger.info('Recall:{}%'.format(r))
    logger.info('F1:{}%'.format(f1))
    logger.info('Validate takes {} s'.format(time.time() - time_valid))

def test():
    time_test = time.time()
    logger.info('loading data...')
    seq_id_test, label_test, input_mask_test, input_segment_test = data_process(
        os.path.join(args.data_path, 'test.txt'), w2i_char, w2i_label, args.max_sequence_len)

    logger.info('building model...')

    bert_config = modeling.BertConfig.from_json_file(args.bert_config_path)
    logger.info(bert_config.to_json_string())

    model = Model(bert_config=bert_config,
                  label_size=len(w2i_label),
                  learning_rate=args.learning_rate)

    with tf.Session(config=tf_config) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt_save_path = tf.train.latest_checkpoint(ckpt_path)
        logger.info('loading checkpoint from {}'.format(ckpt_save_path))
        saver.restore(session, ckpt_save_path)

        predict_label = []
        real_label = []
        p_sum = 0
        r_sum = 0
        hits = 0

        logger.info('start testing ...')

        batch_train = batch_iter(seq_id_test, label_test, input_mask_test, input_segment_test, args.batch_size)
        for i, (inputs_seq_batch, label_batch, input_mask_batch, input_segment_batch) in enumerate(batch_train):
            feed_dict = {model.inputs_seq: inputs_seq_batch, model.label: label_batch,
                         model.inputs_mask: input_mask_batch, model.inputs_segment: input_segment_batch}
            predict_label_prob_batch = session.run(model.probabilities, feed_dict=feed_dict)
            for i in range(len(predict_label_prob_batch)):
                predict_label.append([i for i, prob in enumerate(predict_label_prob_batch[i]) if prob > 0.5])
                real_label.append([i for i, prob in enumerate(label_batch[i]) if prob == 1])

            for predict_label_, real_label_ in zip(predict_label, real_label):
                p_sum += len(predict_label_)
                r_sum += len(real_label_)
                for label in predict_label_:
                    if label in real_label_:
                        hits += 1
        p = hits * 100 / p_sum if p_sum != 0 else 0
        r = hits * 100 / r_sum if r_sum != 0 else 0
        f1 = (2 * p * r) / (p + r) if p + r > 0 else 0
        logger.info('Precision:{}%'.format(p))
        logger.info('Recall:{}%'.format(r))
        logger.info('F1:{}%'.format(f1))
        logger.info('Test takes {} s'.format(time.time() - time_test))


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        logger.info('Please input the right mode: train/test')
