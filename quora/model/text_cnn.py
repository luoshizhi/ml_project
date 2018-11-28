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
SEQ_LEN = config.cf.getint("base", "SEQ_LEN")
#TRAIN_NUM_STEP = config.cf.getint("base", "TRAIN_NUM_STEP")

# HIDDEN_SIZE = config.cf.getint("base", "HIDDEN_SIZE")
# NUM_LAYERS = config.cf.getint("base", "NUM_LAYERS")
EMBEDDING_SIZE = config.cf.getint("base", "EMBEDDING_SIZE")

# MLP_DIMENSION = config.getintlist("base", "MLP_DIMENSION")
FILTER_SIZES = config.getintlist("base", "FILTER_SIZES")
NUM_FILTERS = config.cf.getint("base", "NUM_FILTERS")
POOLING = config.cf.get("base", "POOLING")
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


class TextCNNModel(object):
    def __init__(self, vocab_size, is_trainning, batch_size, embedding_size,
                 filter_sizes, seq_length, l2_reg_lambda=0.0):
        self.vocab_size = vocab_size
        self.is_trainning = is_trainning
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.filter_sizes = filter_sizes
        self.embedding_size = embedding_size
        # l2
        self.l2_reg_lambda = l2_reg_lambda
        self.l2_loss = tf.constant(0.0)

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.targets = tf.placeholder(
            tf.int32, shape=(batch_size, 2), name="targets")

        with tf.variable_scope("input"):
            self._build_input()
        with tf.variable_scope("cnn"):
            self._build_cnn()
        with tf.variable_scope("output"):
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
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size,
                                self.embedding_size, 1, NUM_FILTERS]
                weight = tf.get_variable(
                 name='weight', shape=filter_shape,
                 initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
                bias = tf.get_variable(
                    name="bias", shape=[NUM_FILTERS],
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
                if POOLING == "max":
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.seq_length * 2 - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                if POOLING == "mean":
                    pooled = tf.nn.avg_pool(
                        h,
                        ksize=[1, self.seq_length * 2 - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                pooled_outputs.append(pooled)
        self.num_filters_total = NUM_FILTERS * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool,
                                      [-1, self.num_filters_total])
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

    def _build_fc(self):
        weight = tf.get_variable(
            "weight",
            shape=[self.num_filters_total, 2],
            initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.Variable(tf.constant(0.1, shape=[2]), name="bias")
        self.l2_loss += tf.nn.l2_loss(weight)
        self.l2_loss += tf.nn.l2_loss(bias)
        self.out = tf.nn.xw_plus_b(
                self.h_drop, weight, bias, name="out")


def run_predict(session, model, data, method="predict",
                save_path="./sample_submission.csv"):
    if method == "predict":
        csvfile = open(save_path, 'w')
        writer = csv.writer(csvfile)
        writer.writerow([data.df.columns[0], "is_duplicate"])
    fetches = [model.predict_class]

    if method == "valid":
        fetches.extend([model.loss, model.accuracy])

    generator = data.batch_generator(batch_size=model.batch_size)

    batches_num = 0
    loss = 0.0
    accuracy = 0.0

    for ids, q1, q2, y in generator:
        feed = {model.seq1_inputs: q1,
                model.seq2_inputs: q2,
                model.keep_prob: 1.0
                }
        if method == "valid":
            feed[model.targets] = y
        res = session.run(fetches, feed_dict=feed)
        if method == "valid":
            batches_num += 1
            loss += res[1]
            accuracy += res[2]
        if method == "predict":
            for i in range(len(ids)):
                writer.writerow([ids[i], res[0][i]])

    if method == "predict":
        csvfile.close()
        return (None, None)
    if method == "valid":
        loss = loss / batches_num
        accuracy = accuracy / batches_num
    return loss, accuracy


def run_model(session, is_trainning, train_model,
              valid_model, train_data, valid_data, saver, global_step=0):

    train_fetches = [train_model.loss,
                     train_model.accuracy,
                     train_model.optimizer]
    step = global_step
    statfile = os.path.join(OUTDIR, "stat.txt")
    if (not os.path.exists(statfile)) or os.path.getsize(statfile) == 0:
        with open(statfile, "w") as f:
            info = "step\tloss\ttrain_accuracy\tvalid_accuracy\tsec/{}batches".format(STAT_STEP)
            f.write(info+"\n")
    for i in range(NUM_EPOCH):
        train_batches = train_data.batch_generator(
                        batch_size=train_model.batch_size,
                        shuffle=True, equal=True)
        print("In iteration: %d" % (i + 1))
        start = time.time()
        for _, x1, x2, y in train_batches:
            feed = {train_model.seq1_inputs: x1,
                    train_model.seq2_inputs: x2,
                    train_model.targets: y,
                    train_model.keep_prob: KEEP_PROB}
            train_res = session.run(train_fetches, feed_dict=feed)
            step += 1
            if step % STAT_STEP == 0:
                # print (train_res)
                valid_loss, valid_accuracy = run_predict(
                    session, valid_model, valid_data, method="valid")
                end = time.time()
                print ("step:{0}...".format(step),
                       "loss: {:.4f}...".format(train_res[0]),
                       "train_accuracy:{:.4f}...".format(train_res[1]),
                       "valid_accuracy:{:.4f}...".format(valid_accuracy),
                       "{0:.4f} sec/{1}batches".format(
                                                (end - start), STAT_STEP))
                with open(os.path.join(OUTDIR, "stat.txt"), "a") as f:
                    f.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(
                                                    step,
                                                    train_res[0],
                                                    train_res[1],
                                                    valid_accuracy,
                                                    end - start))
                start = time.time()
        # save model
        saver.save(session,
                   os.path.join(CHECKPOINT_PATH, "model"), global_step=step)
    saver.save(session,
               os.path.join(CHECKPOINT_PATH, "model"), global_step=step)
    f.close()
    return


def main():
    initializer = tf.random_uniform_initializer(-INITIAL, INITIAL)

    # process("../data/train_clean_text.csv", "../data/train_word_to_index10000_0.9.csv","../data/valid_word_to_index10000_0.1.csv")
    # return
    train_data = preprocessing.TextConverter(TRAIN_DATA)
    train_data.cut_padding(cut_size=SEQ_LEN, padding="left")
    train_data.format_inputs()

    valid_data = preprocessing.TextConverter(VALID_DATA)
    valid_data.cut_padding(cut_size=SEQ_LEN, padding="left")
    valid_data.format_inputs()

    test_data = preprocessing.TextConverter(TEST_DATA)
    test_data.cut_padding(cut_size=SEQ_LEN, padding="left")
    test_data.format_inputs()
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        train_model = TextCNNModel(
                                vocab_size=VOCAB_SIZE,
                                is_trainning=True,
                                batch_size=TRAIN_BATCH_SIZE,
                                embedding_size=EMBEDDING_SIZE,
                                seq_length=SEQ_LEN,
                                filter_sizes=FILTER_SIZES,
                                l2_reg_lambda=L2_REG_LAMBDA
                                )
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        valid_model = TextCNNModel(
                                vocab_size=VOCAB_SIZE,
                                is_trainning=False,
                                batch_size=valid_data.len,
                                embedding_size=EMBEDDING_SIZE,
                                seq_length=SEQ_LEN,
                                filter_sizes=FILTER_SIZES,
                                l2_reg_lambda=L2_REG_LAMBDA
                                )
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        predict_model = TextCNNModel(
                                vocab_size=VOCAB_SIZE,
                                is_trainning=False,
                                batch_size=1,
                                embedding_size=EMBEDDING_SIZE,
                                seq_length=SEQ_LEN,
                                filter_sizes=FILTER_SIZES,
                                l2_reg_lambda=L2_REG_LAMBDA
                                )

    start = time.time()
    saver = tf.train.Saver()
    global_step = 0
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        model_file = tf.train.latest_checkpoint(CHECKPOINT_PATH)
        if model_file:
            global_step = int(model_file.split("-")[-1])
            saver.restore(session, model_file)
        if not FLAGS.predict_only:
            run_model(session, True, train_model,
                      valid_model, train_data, valid_data, saver, global_step)
        run_predict(session, predict_model, test_data, method="predict",
                    save_path=os.path.join(OUTDIR, "sample_submission.csv"))
    end = time.time()
    consumption = (end - start)/60/60
    projectname = os.path.basename(OUTDIR)
    plot(os.path.join(OUTDIR, "stat.txt"),
         collist=["loss", "train_accuracy", "valid_accuracy"],
         title=projectname,
         savefig=os.path.join(OUTDIR, projectname+".stat.pdf"))


if __name__ == "__main__":
    main()
