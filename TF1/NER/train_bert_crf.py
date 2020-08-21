# !usr/bin/env python
# -*- coding:utf-8 _*-
# author:chenmeng
# datetime:2020/8/11 14:09
import argparse
import logging
import os
import time

import tensorflow as tf

from bert import modeling
from TF1.NER.eval import conlleval
from TF1.NER.model_bert_crf import ModelBertCRF
from TF1.NER.utils import batch_iter
from TF1.NER.utils import data_process
from TF1.NER.utils import load_vocabulary
from TF1.NER.utils import str2bool

## hyperparameters
parser = argparse.ArgumentParser(description='BERT-CRF for Chinese NER task')

parser.add_argument('--bert_vocab_path', type=str, default='../../bert_ckpt/vocab.txt', help='bert vocab path')
parser.add_argument('--bert_config_path', type=str, default='../../bert_ckpt/bert_config.json', help='bert config path')
parser.add_argument('--bert_ckpt_path', type=str, default='../../bert_ckpt/bert_model.ckpt', help='bert checkpoint path')
parser.add_argument('--data_path', type=str, default='../../data/small_ner',
                    help='data path, this path includes train data test data dev data')
parser.add_argument('--output_path', type=str, default='./output', help='output path')
parser.add_argument('--max_sequence_len', type=int, default=128, help='max length of sequence')
parser.add_argument('--batch_size', type=int, default=100, help='sample of each minibatch')
parser.add_argument('--epoch', type=int, default=10, help='num of training epoch')
parser.add_argument('--use_lstm', type=str2bool, default=False, help='use lstm or not')
parser.add_argument('--use_crf', type=str2bool, default=False, help='use crf or not, if False, use softmax')
parser.add_argument('--num_units', type=int, default=300, help='dim of lstm hidden state')
parser.add_argument('--learning_rate_bert', type=float, default=2e-5, help='bert learning rate')
parser.add_argument('--learning_rate_crf', type=float, default=2e-3, help='crf learning rate')
parser.add_argument('--mode', type=str, default='test', help='train/test')
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
w2i_bio, i2w_bio = load_vocabulary(os.path.join(args.data_path, 'vocab_bio.txt'))


def train():
    logger.info('loading data...')
    seq_id_train, label_id_train, input_mask_train, input_segment_train = data_process(
        os.path.join(args.data_path, 'train.txt'), w2i_char, w2i_bio, args.max_sequence_len)
    seq_id_val, label_id_val, input_mask_val, input_segment_val = data_process(
        os.path.join(args.data_path, 'dev.txt'), w2i_char, w2i_bio, args.max_sequence_len)

    logger.info('building model...')

    bert_config = modeling.BertConfig.from_json_file(args.bert_config_path)
    logger.info(bert_config.to_json_string())

    model = ModelBertCRF(bert_config=bert_config, vocab_size_bio=len(w2i_bio),
                         use_lstm=args.use_lstm, use_crf=args.use_crf, num_units=args.num_units,
                         max_seq_len=args.max_sequence_len,
                         learning_rate_bert=args.learning_rate_bert, learning_rate_crf=args.learning_rate_crf)

    logger.info("model params:")
    params_num_all = 0
    for variable in tf.trainable_variables():
        params_num = 1
        for dim in variable.shape:
            params_num *= dim
        params_num_all += params_num
        logger.info("\t {} {} {}".format(variable.name, variable.shape, params_num))
    logger.info("all params num: " + str(params_num_all))

    logger.info('loading bert pretrained parameters...')

    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                               args.bert_ckpt_path)
    tf.train.init_from_checkpoint(args.bert_ckpt_path, assignment_map)

    logger.info('configuring tensorboard...')
    tf.summary.scalar('loss', model.loss)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_path)

    logger.info('start training...')

    # 创建session
    with tf.Session(config=tf_config) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3)
        writer.add_graph(session.graph)

        losses = []
        batches = 0
        for epoch in range(args.epoch):
            batch_train = batch_iter(seq_id_train, label_id_train, input_mask_train, input_segment_train,
                                     args.batch_size)
            for inputs_seq_batch, outputs_seq_batch, inputs_mask_batch, inputs_segment_batch in batch_train:
                if batches == 0:
                    logger.info("###### shape of a batch #######")
                    logger.info("inputs_seq: " + str(inputs_seq_batch.shape))
                    logger.info("inputs_mask: " + str(inputs_mask_batch.shape))
                    logger.info("inputs_segment: " + str(inputs_segment_batch.shape))
                    logger.info("outputs_seq: " + str(outputs_seq_batch.shape))
                    logger.info("###### preview a sample #######")
                    logger.info("input_seq:" + " ".join([i2w_char[i] for i in inputs_seq_batch[0]]))
                    logger.info("input_mask :" + " ".join([str(i) for i in inputs_mask_batch[0]]))
                    logger.info("input_segment :" + " ".join([str(i) for i in inputs_segment_batch[0]]))
                    logger.info("output_seq: " + " ".join([i2w_bio[i] for i in outputs_seq_batch[0]]))
                    logger.info("###############################")
                feed_dict = {model.inputs_seq: inputs_seq_batch, model.outputs_seq: outputs_seq_batch,
                             model.inputs_mask: inputs_mask_batch, model.inputs_segment: inputs_segment_batch}
                loss, _ = session.run([model.loss, model.train_op], feed_dict)
                losses.append(loss)

                if batches % 100 == 0:
                    logger.info('epoch: {}'.format(epoch))
                    logger.info('batch: {}'.format(batches))
                    logger.info('loss: {}'.format(sum(losses) / len(losses)))
                    losses = []

                    s = session.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, batches)

                    # 在验证集进行验证，并保存FB1最高的模型
                    valid(model, session, seq_id_val, label_id_val, input_mask_val, input_segment_val, epoch)
                    ckpt_save_path = os.path.join(ckpt_path, 'model.ckpt')
                    logger.info('path of ckpt: {}-{}'.format(ckpt_save_path, batches))
                    saver.save(session, ckpt_save_path, global_step=batches)
                batches += 1


def valid(model, sess, seq_id_val, label_id_val, input_mask_val, input_segment_val, epoch_num):
    batch_train = batch_iter(seq_id_val, label_id_val, input_mask_val, input_segment_val, args.batch_size)
    model_predict = []
    for inputs_seq_batch, output_seq_batch, input_mask_batch, input_segment_batch in batch_train:
        feed_dict = {model.inputs_seq: inputs_seq_batch, model.outputs_seq: output_seq_batch,
                     model.inputs_mask: input_mask_batch, model.inputs_segment: input_segment_batch}
        predict_seq_batch = sess.run(model.outputs, feed_dict=feed_dict)
        # model_predict = [ [[char, real_label, predict_label_id]   ]  ]
        for i in range(len(predict_seq_batch)):
            predict_single = []
            for seq_id, label_id, predict_id in zip(inputs_seq_batch[i], output_seq_batch[i], predict_seq_batch[i]):
                if i2w_char[seq_id] == '[CLS]':
                    continue
                if i2w_char[seq_id] == '[SEP]':
                    break
                predict_single.append(
                    [i2w_char[seq_id], i2w_bio[label_id], i2w_bio[predict_id] if predict_id != 0 else 0])
            model_predict.append(predict_single)

    label_path = os.path.join(result_path, 'label_' + str(epoch_num))
    metric_path = os.path.join(result_path, 'result_metric_' + str(epoch_num))
    # value_overall = {}
    # flag = True
    for _ in conlleval(model_predict, label_path, metric_path):
        logger.info(_)
    #     if 'FB1' in _ and flag:
    #         value_list = _.split(' ')
    #         value_overall['FB1'] = float(value_list[-1])
    #         flag = False
    # return value_overall


def test():
    logger.info('loading data...')
    seq_id_test, label_id_test, input_mask_test, input_segment_test = data_process(
        os.path.join(args.data_path, 'test.txt'), w2i_char, w2i_bio, args.max_sequence_len)

    logger.info('building model...')

    bert_config = modeling.BertConfig.from_json_file(args.bert_config_path)
    logger.info(bert_config.to_json_string())

    model = ModelBertCRF(bert_config=bert_config, vocab_size_bio=len(w2i_bio),
                         use_lstm=args.use_lstm, use_crf=args.use_crf, num_units=args.num_units,
                         max_seq_len=args.max_sequence_len,
                         learning_rate_bert=args.learning_rate_bert, learning_rate_crf=args.learning_rate_crf)

    with tf.Session(config=tf_config) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt_save_path = tf.train.latest_checkpoint(ckpt_path)
        logger.info('loading checkpoint from {}'.format(ckpt_save_path))
        saver.restore(session, ckpt_save_path)

        logger.info('start test...')

        valid(model, session, seq_id_test, label_id_test, input_mask_test, input_segment_test, 'test')


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
