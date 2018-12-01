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

    train_data = preprocessing.TextConverter(TRAIN_DATA)
    train_data.cut_padding(cut_size=SEQ_LEN, padding=PADDING)
    train_data.format_inputs()

    valid_data = preprocessing.TextConverter(VALID_DATA)
    valid_data.cut_padding(cut_size=SEQ_LEN, padding=PADDING)
    valid_data.format_inputs()

    test_data = preprocessing.TextConverter(TEST_DATA)
    test_data.cut_padding(cut_size=SEQ_LEN, padding=PADDING)
    test_data.format_inputs()
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        train_model = MPCNNModel(
                                vocab_size=VOCAB_SIZE,
                                is_trainning=True,
                                batch_size=TRAIN_BATCH_SIZE,
                                embedding_size=EMBEDDING_SIZE,
                                num_filters=NUM_FILTERS,
                                seq_length=SEQ_LEN,
                                h_pool_type=H_POOL_TYPE,
                                h_ws_sizes=H_WS_SIZE,
                                p_pool_type=P_POOL_TYPE,
                                p_ws_sizes=P_WS_SIZE,
                                l2_reg_lambda=L2_REG_LAMBDA
                                )
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        valid_model = MPCNNModel(
                                vocab_size=VOCAB_SIZE,
                                is_trainning=False,
                                batch_size=valid_data.len,
                                embedding_size=EMBEDDING_SIZE,
                                num_filters=NUM_FILTERS,
                                seq_length=SEQ_LEN,
                                h_pool_type=H_POOL_TYPE,
                                h_ws_sizes=H_WS_SIZE,
                                p_pool_type=P_POOL_TYPE,
                                p_ws_sizes=P_WS_SIZE,
                                l2_reg_lambda=L2_REG_LAMBDA
                                )
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        predict_model = MPCNNModel(
                                vocab_size=VOCAB_SIZE,
                                is_trainning=False,
                                batch_size=1,
                                embedding_size=EMBEDDING_SIZE,
                                num_filters=NUM_FILTERS,
                                seq_length=SEQ_LEN,
                                h_pool_type=H_POOL_TYPE,
                                h_ws_sizes=H_WS_SIZE,
                                p_pool_type=P_POOL_TYPE,
                                p_ws_sizes=P_WS_SIZE,
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
