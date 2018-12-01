# -*- coding: utf-8 -*-
from plot import plot
import tensorflow as tf
import preprocessing
from preprocessing import ConfigFile
import time
import sys
import csv
import os


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_boolean('predict_only', False, 'only prdict not re_training')
config = ConfigFile(sys.argv[1])

TRAIN_DATA = config.cf.get("base", "TRAIN_DATA")
VALID_DATA = config.cf.get("base", "VALID_DATA")
TEST_DATA = config.cf.get("base", "TEST_DATA")
PADDING = config.cf.get("base", "PADDING")
TRAIN_BATCH_SIZE = config.cf.getint("base", "TRAIN_BATCH_SIZE")
VOCAB_SIZE = config.cf.getint("base", "VOCAB_SIZE")
SEQ_LEN = config.cf.getint("base", "SEQ_LEN")
#TRAIN_NUM_STEP = config.cf.getint("base", "TRAIN_NUM_STEP")

# HIDDEN_SIZE = config.cf.getint("base", "HIDDEN_SIZE")
# NUM_LAYERS = config.cf.getint("base", "NUM_LAYERS")
EMBEDDING_SIZE = config.cf.getint("base", "EMBEDDING_SIZE")

# MLP_DIMENSION = config.getintlist("base", "MLP_DIMENSION")
FILTER_SIZES = config.getintlist("base", "FILTER_SIZES")
NUM_FILTERS = config.cf.getint("base", "NUM_FILTERS")
POOLING = config.cf.get("base", "POOLING")
CNNTYPE = config.cf.get("base", "CNNTYPE")

STAT_STEP = config.cf.getint("base", "STAT_STEP")

NUM_EPOCH = config.cf.getint("base", "NUM_EPOCH")
KEEP_PROB = config.cf.getfloat("base", "KEEP_PROB")
# EMBEDDING_KEEP_PROB = config.cf.getfloat("base", "EMBEDDING_KEEP_PROB")
MAX_GRAD_NORM = config.cf.getint("base", "MAX_GRAD_NORM")
LEARNING_RATE = config.cf.getfloat("base", "LEARNING_RATE")
EVAL_BATCH_SIZE = config.cf.getint("base", "EVAL_BATCH_SIZE")

# OUTDIR = config.cf.get("base", "OUTDIR")
OUTDIR = os.path.dirname(os.path.abspath(sys.argv[1]))
CHECKPOINT_PATH = os.path.join(OUTDIR, "cnn_model")
INITIAL = config.cf.getfloat("base", "INITIAL")
L2_REG_LAMBDA = config.cf.getfloat("base", "L2_REG_LAMBDA")


class MPCNNModel(object):
    def __init__(self, vocab_size, is_trainning, batch_size, embedding_size,
                 num_filters, seq_length, h_pool_type, p_pool_type,
                 h_ws_sizes, p_ws_sizes, l2_reg_lambda=0.0):
        self.vocab_size = vocab_size
        self.is_trainning = is_trainning
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_filters = num_filters
        self.h_pool_type = h_pool_type
        self.p_pool_type = p_pool_type
        self.h_ws_sizes = h_ws_sizes
        self.p_ws_sizes = p_ws_sizes
        self.embedding_size = embedding_size
        # l2
        self.l2_reg_lambda = l2_reg_lambda
        self.l2_loss = tf.constant(0.0)

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.targets = tf.placeholder(
            tf.int32, shape=(batch_size, 2), name="targets")

        with tf.name_scope("input"):
            self.seq1_inputs = tf.placeholder(
                tf.int32, [self.batch_size, self.seq_length], name="seq1")
            self.seq2_inputs = tf.placeholder(
                tf.int32, [self.batch_size, self.seq_length], name="seq2")
            s1, s2 = self._build_input(self.seq1_inputs, self.seq2_inputs)

        with tf.variable_scope("blockA"):
            m1_pool_h = self._build_block(s1,
                                          self.h_pool_type,
                                          self.h_ws_sizes,
                                          block="holistic")
            m2_pool_h = self._build_block(s2,
                                          self.h_pool_type,
                                          self.h_ws_sizes,
                                          block="holistic")
            feah_h, feab_h = self._cal_block(m1_pool_h,
                                             m2_pool_h,
                                             block="holistic")

        with tf.variable_scope("blockB"):
            m1_pool_p = self._build_block(s1,
                                          self.p_pool_type,
                                          self.p_ws_sizes,
                                          block="per_dimension")
            m2_pool_p = self._build_block(s2,
                                          self.p_pool_type,
                                          self.p_ws_sizes,
                                          block="per_dimension")
            feah_p, feab_p = self._cal_block(m1_pool_p,
                                             m2_pool_p,
                                             block="per_dimension")
        self.cnn_out = tf.concat([feah_h, feab_h, feah_p, feab_p], axis=1)

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.cnn_out, self.keep_prob)

        with tf.name_scope("output"):
            self._build_fc()

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=self.targets, logits=self.out))
            self.loss += self.l2_reg_lambda * self.l2_loss

        with tf.name_scope("assessment"):
            self.predict_class = tf.argmax(self.out, 1)
            self.target_class = tf.argmax(self.targets, 1)
            correct_prediction = tf.equal(self.predict_class,
                                          self.target_class)
            self.accuracy = tf.reduce_mean(
                            tf.cast(correct_prediction, tf.float32))
        if not is_trainning:
            return

        with tf.name_scope("optimzer"):
            trainable_variables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                        tf.gradients(self.loss, trainable_variables),
                        MAX_GRAD_NORM)
            optimizer = tf.train.GradientDescentOptimizer(
                        learning_rate=LEARNING_RATE)
            self.optimizer = optimizer.apply_gradients(
                        zip(grads, trainable_variables))

    def _build_input(self, input1, input2):
        self.embedding = tf.get_variable(
                    'embedding', [self.vocab_size, self.embedding_size])
        s1 = tf.nn.embedding_lookup(self.embedding, input1)
        s2 = tf.nn.embedding_lookup(self.embedding, input2)
        s1_expand = tf.expand_dims(s1, -1)
        s2_expand = tf.expand_dims(s2, -1)
        return s1_expand, s2_expand

    def _build_block(self, seq_expand, pool_type,
                     ws_sizes, block="holistic"):
        """block="holistic" or "per_dimension"
        """
        m_pool = []
        for i, pool in enumerate(pool_type):
            ws_pool = []
            for j, ws in enumerate(ws_sizes):
                scope_name = "{}-conv-{}pool-{}".format(block, pool, ws)
                with tf.name_scope(scope_name):
                    # conv layers
                    if block == "holistic":
                        filter_shape = [ws, self.embedding_size,
                                        1, self.num_filters]
                    if block == "per_dimension":
                        filter_shape = [ws, 1,
                                        1, self.num_filters]
                    weight = tf.Variable(tf.truncated_normal(
                        filter_shape, stddev=0.1), name="weight")
                    conv_s1 = tf.nn.conv2d(seq_expand, weight,
                                           strides=[1, 1, 1, 1],
                                           padding="VALID", name="conv")
                    # pool layers
                    ksize = [1, self.seq_length - ws + 1, 1, 1]
                    if pool == 'max':
                        pool_out = tf.nn.max_pool(
                            conv_s1, ksize=ksize, strides=[1, 1, 1, 1],
                            padding='VALID', name="pool")
                    elif pool == 'mean':
                        pool_out = tf.nn.avg_pool(
                            conv_s1, ksize, strides=[1, 1, 1, 1],
                            padding='VALID', name='pool')
                    else:
                        pool_out = -tf.nn.max_pool(
                            -conv_s1, ksize, strides=[1, 1, 1, 1],
                            padding='VALID', name="pool")
                    pool_out_expand = tf.transpose(
                                        tf.squeeze(pool_out, [1]), [0, 2, 1])
                    ws_pool.append(pool_out_expand)
            m_pool.append(ws_pool)
        return tf.concat(m_pool, 3)

    def _cal_block(self, m1_pool, m2_pool, block="holistic"):
        # for holistic:split 0:max, 1:mean, 2:min
        # for per_dimension:split 0:max, 1:mean
        split1 = [tf.squeeze(sp, [0])
                  for sp in tf.split(m1_pool, m1_pool.shape.as_list()[0], 0)]
        split2 = [tf.squeeze(sp, [0])
                  for sp in tf.split(m2_pool, m2_pool.shape.as_list()[0], 0)]
        cal_feah = []
        cal_feab = []
        for i in range(len(split1)):
            # calculate feah holistic or per_dimension
            cal_sub_feah = tf.reduce_sum(tf.multiply(split1[i], split2[i],
                                         name="cal_feah_"+block+str(i)),
                                         axis=2)
            # calculate feab holistic or per_dimension
            cal_sub_feab = tf.reduce_sum(tf.multiply(split1[i], split2[i],
                                         name="cal_feab_"+block+str(i)),
                                         axis=1)
            cal_feah.append(cal_sub_feah)
            cal_feab.append(cal_sub_feab)
        feah = tf.concat(cal_feah, axis=1, name="feah_"+block+"_concat")
        feab = tf.concat(cal_feab, axis=1, name="feab_"+block+"_concat")

        return feah, feab

    def _build_fc(self):
        weight = tf.get_variable(
            "weight",
            shape=[self.cnn_out.shape.as_list()[1], 2],
            initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.Variable(tf.constant(0.1, shape=[2]), name="bias")
        self.l2_loss += tf.nn.l2_loss(weight)
        self.l2_loss += tf.nn.l2_loss(bias)
        self.out = tf.nn.xw_plus_b(
                self.h_drop, weight, bias, name="out")
