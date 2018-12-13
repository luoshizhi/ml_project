# -*- coding: utf-8 -*-
from plot import plot
import tensorflow as tf
import inspect
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
H_POOL_TYPE = config.getlist("base", "H_POOL_TYPE")
H_WS_SIZE = config.getintlist("base", "H_WS_SIZE")
P_POOL_TYPE = config.getlist("base", "P_POOL_TYPE")
P_WS_SIZE = config.getintlist("base", "P_WS_SIZE")


EMBEDDING_SIZE = config.cf.getint("base", "EMBEDDING_SIZE")

NUM_FILTERS = config.cf.getint("base", "NUM_FILTERS")

STAT_STEP = config.cf.getint("base", "STAT_STEP")

NUM_EPOCH = config.cf.getint("base", "NUM_EPOCH")
KEEP_PROB = config.cf.getfloat("base", "KEEP_PROB")
MAX_GRAD_NORM = config.cf.getint("base", "MAX_GRAD_NORM")
LEARNING_RATE = config.cf.getfloat("base", "LEARNING_RATE")
EVAL_BATCH_SIZE = config.cf.getint("base", "EVAL_BATCH_SIZE")

OUTDIR = os.path.dirname(os.path.abspath(sys.argv[1]))
CHECKPOINT_PATH = os.path.join(OUTDIR, "mpcnn_model")
INITIAL = config.cf.getfloat("base", "INITIAL")
L2_REG_LAMBDA = config.cf.getfloat("base", "L2_REG_LAMBDA")


class DecompModel(object):
    def __init__(self, keep_prob):
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.l2 = l2
        self.clip_value = clip_value

        with tf.name_scope("input"):
            self.seq1_inputs = tf.placeholder(
                tf.int32, [self.batch_size, self.seq_length], name="seq1")
            self.seq2_inputs = tf.placeholder(
                tf.int32, [self.batch_size, self.seq_length], name="seq2")
            s1, s2 = self._build_input(self.seq1_inputs, self.seq2_inputs)

    def _build_input(self, input1, input2):
        self.embedding = tf.get_variable(
                    'embedding', [self.vocab_size, self.embedding_size])
        s1 = tf.nn.embedding_lookup(self.embedding, input1)
        s2 = tf.nn.embedding_lookup(self.embedding, input2)
        return s1, s2

    def _feed_forward(self, inputs, out_units, num_layers=2, reuse=False,
                      initializer=None):
        if initializer is None:
            initializer = tf.contrib.layers.xavier_initializer()
        for i in range(num_layers):
            with tf.variable_scope("feed_forward"+str(i+1)):
                inputs = tf.nn.dropout(inputs, self.keep_prob)
                output = tf.layers.dense(inputs, out_units, tf.nn.relu,
                                         kernel_initializer=initializer)
        return output

    def _attend(self, s1, s2):
        F_a_bar  = self._feedForwardBlock(s1, self.hidden_size)



