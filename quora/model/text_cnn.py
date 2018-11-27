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
TRAIN_BATCH_SIZE = config.cf.getint("base", "TRAIN_BATCH_SIZE")
VOCAB_SIZE = config.cf.getint("base", "VOCAB_SIZE")
TRAIN_NUM_STEP = config.cf.getint("base", "TRAIN_NUM_STEP")
# HIDDEN_SIZE = config.cf.getint("base", "HIDDEN_SIZE")
# NUM_LAYERS = config.cf.getint("base", "NUM_LAYERS")
EMBEDDING_SIZE = config.cf.getint("base", "EMBEDDING_SIZE")

MLP_DIMENSION = config.getintlist("base", "MLP_DIMENSION")
# STAT_STEP = config.cf.getint("base", "STAT_STEP")

NUM_EPOCH = config.cf.getint("base", "NUM_EPOCH")
KEEP_PROB = config.cf.getfloat("base", "KEEP_PROB")
EMBEDDING_KEEP_PROB = config.cf.getfloat("base", "EMBEDDING_KEEP_PROB")
MAX_GRAD_NORM = config.cf.getint("base", "MAX_GRAD_NORM")
LEARNING_RATE = config.cf.getfloat("base", "LEARNING_RATE")
EVAL_BATCH_SIZE = config.cf.getint("base", "EVAL_BATCH_SIZE")

# OUTDIR = config.cf.get("base", "OUTDIR")
OUTDIR = os.path.dirname(os.path.abspath(sys.argv[1]))
CHECKPOINT_PATH = os.path.join(OUTDIR, "cnn_model")
INITIAL = config.cf.getfloat("base", "INITIAL")


class TextCNNModel(object):
    def __init__(self, vocab_size, is_trainning, batch_size, embedding_size,
                 filter_sizes, seq_length):
        self.vocab_size = vocab_size
        self.is_trainning = is_trainning
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.filter_sizes = filter_sizes
        self.embedding_size = embedding_size
        # l2
        # self.l2_reg_lambda = l2_reg_lambda
        # self.l2_loss = tf.constant(0.0)

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.targets = tf.placeholder(
            tf.int32, shape=(batch_size, 2), name="targets")

        with tf.variable_scope("input"):
            self._build_input()
        with tf.variable_scope("cnn"):
            self._build_cnn()
        with tf.variable_scope("output"):
            self._buil_fc()
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=self.targets, logits=self.scores))
        with tf.name_scope("assessment"):
            self.predict_class = tf.argmax(self.scores, 1)
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

    def _build_input(self):
        self.embedding = tf.get_variable(
                    'embedding', [self.vocab_size, self.embedding_size])
        self.seq1_inputs = tf.placeholder(
            tf.int32, [self.batch_size, self.seq_length], name="seq1")
        self.seq2_inputs = tf.placeholder(
            tf.int32, [self.batch_size, self.seq_length], name="seq2")
        self.s1 = tf.nn.embedding_lookup(self.embedding, self.seq1_inputs)
        self.s2 = tf.nn.embedding_lookup(self.embedding, self.seq2_inputs)
        inputs = tf.concat([self.s1, self.s2], axis=1)
        self.inputs = tf.expand_dims(inputs, -1)

    def _build_cnn(self):
        pooled_outputs = []
        num_filters = len(self.filter_sizes)
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size,
                                self.embedding_size, 1, num_filters]
                weight = tf.get_variable(
                 name='weight', shape=filter_shape,
                 initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
                bias = tf.get_variable(
                    name="bias", shape=[num_filters],
                    initializer=tf.ones_initializer())
                conv = tf.nn.conv2d(
                    self.inputs,
                    weight,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length * 2 - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        self.num_filters_total = num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool,
                                      [-1, self.num_filters_total])

    def _build_fc(self):
        weight = tf.get_variable(
            "weight",
            shape=[self.num_filters_total, 2],
            initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.Variable(tf.constant(0.1, shape=[2]), name="bias")
        # l2_loss += tf.nn.l2_loss(W)
        # l2_loss += tf.nn.l2_loss(b)
        self.scores = tf.nn.xw_plus_b(
                self.h_drop, weight, bias, name="scores")

