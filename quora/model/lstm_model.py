# -*- coding: utf-8 -*-
import tensorflow as tf
import inspect
import preprocessing
from preprocessing import ConfigFile
import time
import sys

config = ConfigFile(sys.argv[1])

TRAIN_DATA = config.cf.get("base", "TRAIN_DATA")
VALID_DATA = config.cf.get("base", "VALID_DATA")
TEST_DATA = config.cf.get("base", "TEST_DATA")
TRAIN_BATCH_SIZE = config.cf.getint("base", "TRAIN_BATCH_SIZE")
VOCAB_SIZE = config.cf.getint("base", "VOCAB_SIZE")
TRAIN_NUM_STEP = config.cf.getint("base", "TRAIN_NUM_STEP")
HIDDEN_SIZE = config.cf.getint("base", "HIDDEN_SIZE")
NUM_LAYERS = config.cf.getint("base", "NUM_LAYERS")
MLP_DIMENSION = config.getintlist("base", "MLP_DIMENSION")


NUM_EPOCH = config.cf.getint("base", "NUM_EPOCH")
KEEP_PROB = config.cf.getfloat("base", "KEEP_PROB")
EMBEDDING_KEEP_PROB = config.cf.getfloat("base", "EMBEDDING_KEEP_PROB")
MAX_GRAD_NORM = config.cf.getint("base", "MAX_GRAD_NORM")
LEARNING_RATE = config.cf.getfloat("base", "LEARNING_RATE")
EVAL_BATCH_SIZE = config.cf.getint("base", "EVAL_BATCH_SIZE")


class QuoraModel(object):
    def __init__(self, vocab_size, is_trainning, batch_size, num_step,
                 hidden_size, num_layers):
        self.vocab_size = vocab_size
        self.is_trainning = is_trainning
        self.num_step = num_step
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        # 测试时采用reuse=True ，可以使用训练时的参数
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.embedding = tf.get_variable(
                        'embedding', [self.vocab_size, self.hidden_size])
        self.targets = tf.placeholder(
                        tf.int32, shape=(batch_size, 2), name="targets")

        with tf.variable_scope("seq1"):
            self.seq1_inputs, self.seq1_emb_inputs = self._build_inputs()
            (self.seq1_rnn_outputs, self.seq1_final_state,
             self.seq1_initial_state) = self._build_rnn(
                                        self.seq1_emb_inputs, model="lstm")

        with tf.variable_scope("seq2"):
            self.seq2_inputs, self.seq2_emb_inputs = self._build_inputs()
            (self.seq2_rnn_outputs, self.seq2_final_state,
             self.seq2_initial_state) = self._build_rnn(
                                        self.seq2_emb_inputs, model="lstm")

        with tf.name_scope("merge"):
            # [batch_size, hidden_size * 2]
            self.out_merge = tf.concat([self.seq1_rnn_outputs,
                                        self.seq2_rnn_outputs], axis=1)

        with tf.variable_scope("mlp"):
            self.predict = self._build_mlp(self.out_merge, MLP_DIMENSION)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=self.targets, logits=self.predict))

        with tf.name_scope("assessment"):
            self.predict_class = tf.argmax(self.predict, 1)
            self.target_class = tf.argmax(self.targets, 1)
            correct_prediction = tf.equal(self.predict_class,
                                          self.target_class)
            self.accuracy = tf.reduce_mean(
                            tf.cast(correct_prediction, tf.float32))
            # tf.where(predict_class)
            # self.precision =
            # self.recall =
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

    def _build_inputs(self):
        inputs = tf.placeholder(tf.int32,
                                shape=(self.batch_size, self.num_step),
                                name="inputs")
        emb_inputs = tf.nn.embedding_lookup(self.embedding, inputs)
        return inputs, emb_inputs

    def _build_rnn(self, emb_inputs, model="lstm",):
        def get_cell():
            # debug 这个报错的原因是tensorflow v1.1.0以上版本的BasicLSTMCell加入了reuse参数，默认是False,
            # 但是该reuse参数不会随全局tf.get_variable_scope().reuse变化，因此需要手动加上
            if model == "lstm":
                if 'reuse' in inspect.signature(
                                tf.contrib.rnn.LSTMCell.__init__).parameters:
                    rnn = tf.contrib.rnn.LSTMCell(
                            self.hidden_size,
                            reuse=tf.get_variable_scope().reuse)
                else:
                    return tf.contrib.rnn.LSTMCell(self.hidden_size)
            if model == "rnn":
                rnn = tf.contrib.rnn.BasicRNNCell(self.hidden_size)
            if model == "gru":
                rnn = tf.contrib.rnn.GRUCell(self.hidden_size)
            cell = tf.contrib.rnn.DropoutWrapper(
                    rnn, output_keep_prob=self.keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
                [get_cell() for _ in range(self.num_layers)])

        initial_state = cell.zero_state(self.batch_size, tf.float32)
        rnn_output, final_state = tf.nn.dynamic_rnn(
                        cell, emb_inputs, initial_state=initial_state)
        # final_stat[-1].h or tf.transpose(rnn_output,[1,0,2])[-1]
        return (tf.transpose(rnn_output, [1, 0, 2])[-1],
                final_state,
                initial_state)

    def _build_mlp(self, inputs, layer_dimension,
                   activate=tf.nn.relu, use_dropout=False, keep_prob=0.5):
        layer_dimension = [inputs.shape.as_list()[1]] + layer_dimension
        cur_layer = inputs
        self.Wx_plus_b = []
        self.layers = []

        def dropout_or_not(Wx_plus_b, keep_prob, use=False):
            if use is True:
                return tf.nn.dropout(Wx_plus_b, keep_prob)
            else:
                return Wx_plus_b
        # L1, L2 or not?
        # dropout or not?
        n_layers = len(layer_dimension)
        in_dim = layer_dimension[0]
        for i in range(1, n_layers):
            if i == (n_layers - 1):
                use_dropout = False
            out_dim = layer_dimension[i]
            with tf.variable_scope("layer"+str(i)):
                weight = tf.get_variable('weight', [in_dim, out_dim])
                bias = tf.get_variable('bias', [out_dim])
                Wx_plus_b = tf.matmul(cur_layer, weight) + bias
                if i != (n_layers - 1):
                    cur_layer = activate(dropout_or_not(Wx_plus_b,
                                                        use=use_dropout,
                                                        keep_prob=1))
                else:
                    cur_layer = Wx_plus_b
                self.Wx_plus_b.append(Wx_plus_b)
                self.layers.append(cur_layer)
            in_dim = layer_dimension[i]
        return cur_layer

'''
def run_epoch(session, model, batches, keep_prob, eval_op=None, verbose=True):
    # fetches = [model.loss, model.accuracy, model.predict,
    #           model.predict_class,  model.target_class, model.out_merge,
    #            model.Wx_plus_b, model.layers]
    fetches = [model.loss, model.accuracy]
    seq1_state = session.run(model.seq1_initial_state)
    seq2_state = session.run(model.seq2_initial_state)
    step = 0
    if eval_op is not None:
        fetches.append(eval_op)
    start = time.time()
    for x1, x2, y in batches:
        feed = {model.seq1_inputs: x1,
                model.seq2_inputs: x2,
                model.targets: y,
                model.keep_prob: keep_prob,
                model.seq1_initial_state: seq1_state,
                model.seq2_initial_state: seq2_state}
        res = session.run(fetches, feed_dict=feed)

        if verbose is True and step % 100 == 0:
            end = time.time()
            print ("step:{0}...".format(step),
                   "loss: {:.4f}...".format(res[0]),
                   "accuracy:{:.4f}...".format(res[1]),
                   # "predict:{0}...".format(res[2]),
                   # "predict_class{0}...".format(res[3]),
                   # "target_class{0}...".format(res[4]),
                   # "out_merge{0}...".format(res[5]),
                   # "Wx_plus_b{0}...".format(res[6]),
                   # "layer{0}...".format(res[7]),
                   "{:.4f} sec/100 batches".format((end - start)),)
            start = time.time()
#            return res
    return res
'''


def run_predict(session, model, data, method="predict",
                save_path=""):
    seq1_state = seq2_state = session.run(
                            model.seq1_initial_state)
    if method == "predict":
        fetches = [model.predict_class]
    elif method == "valid":
        fetches = [model.loss, model.accuracy]
    generator = data.batch_generator(batch_size=model.batch_size)
    batches_num = 0
    loss = 0.0
    accuracy = 0.0
    start = time.time()
    aa=0
    for x in generator:
        aa += 1
        feed = {model.seq1_inputs: x[0],
                model.seq2_inputs: x[1],
                model.keep_prob: 1.0,
                model.seq1_initial_state:
                seq1_state,
                model.seq2_initial_state:
                seq2_state}
        print (x[0].shape)
        if method == "valid":
            feed[model.targets] = x[2]
        res = session.run(fetches, feed_dict=feed)
        print (aa)
        if method == "valid":
            batches_num += 1
            loss += res[0]
            accuracy += res[1]
        end = time.time()
        print (res, end-start)
        start = time.time()
    if method == "predict":
        return (None, None)
        pass
    if method == "valid":
        loss = loss / batches_num
        accuracy = accuracy / batches_num
    return loss, accuracy


def run_model(session, is_trainning, train_model,
              valid_model, train_data, valid_data):
    train_model_seq1_state = train_model_seq2_state = session.run(
                            train_model.seq1_initial_state)
    train_fetches = [train_model.loss,
                     train_model.accuracy,
                     train_model.optimizer]
    step = 0
    for i in range(NUM_EPOCH):
        train_batches = train_data.batch_generator(
                        batch_size=train_model.batch_size, equal=True)
        print("In iteration: %d" % (i + 1))
        start = time.time()
        for x1, x2, y in train_batches:
            feed = {train_model.seq1_inputs: x1,
                    train_model.seq2_inputs: x2,
                    train_model.targets: y,
                    train_model.keep_prob: KEEP_PROB,
                    train_model.seq1_initial_state:
                    train_model_seq1_state,
                    train_model.seq2_initial_state:
                    train_model_seq2_state}
            train_res = session.run(train_fetches, feed_dict=feed)
            step += 1
            #print (train_res)
            if step % 20 == 0:
                print (train_res)
                #"""
                valid_loss, valid_accuracy = run_predict(
                    session, valid_model, valid_data, method="valid")
                end = time.time()
                print ("step:{0}...".format(step),
                       "loss: {:.4f}...".format(train_res[0]),
                       "train_accuracy:{:.4f}...".format(train_res[1]),
                       "test_accuracy:{:.4f}...".format(valid_accuracy),
                       "{:.4f} sec/100 batches".format((end - start)),)
                start = time.time()
                #"""


def process(file, save_train, save_valid, valid_ratio=0.1):
    train_data = preprocessing.TextConverter(file)
    train_data.build_word_index(
            vocab_size=VOCAB_SIZE,
            save_path="../data/train.vocab"+str(VOCAB_SIZE))
    train_data.tran_word_to_index()
    train_data.save("../data/train_word_to_index" + str(VOCAB_SIZE) + ".csv")
    train_data.df = train_data.df.sample(frac=1.0).reset_index(drop=True)
    train_data.valid_df = train_data.df.iloc[:int(
                                        len(train_data.df)*valid_ratio), :]
    train_data.train_df = train_data.df.iloc[int(
                                        len(train_data.df)*valid_ratio):, :]
    train_data.train_df.to_csv(save_train, index=False)
    train_data.valid_df.to_csv(save_valid, index=False)
    return


def main():
    initializer = tf.random_uniform_initializer(-1.0, 1.0)

    #process("../data/train_clean_text.csv", "../data/train_word_to_index10000_0.9.csv","../data/valid_word_to_index10000_0.1.csv")
    #return
    train_data = preprocessing.TextConverter(TRAIN_DATA)
    train_data.cut_padding(cut_size=TRAIN_NUM_STEP, padding="left")
    train_data.format_inputs()

    valid_data = preprocessing.TextConverter(VALID_DATA)
    valid_data.cut_padding(cut_size=TRAIN_NUM_STEP, padding="left")
    valid_data.format_inputs()

    test_data = preprocessing.TextConverter(TEST_DATA)
    test_data.load_word_index(file_path="../data/train.vocab")
    test_data.preprocess()
    test_data.save("../data/test_redu_clean_text.csv")
    test_data.tran_word_to_index()
    test_data.save("../data/test_redu_word_to_index.csv")
    test_data.cut_padding(cut_size=TRAIN_NUM_STEP, padding="left")
    test_data.format_inputs()
    return
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        train_model = QuoraModel(
                                vocab_size=VOCAB_SIZE,
                                is_trainning=True,
                                batch_size=TRAIN_BATCH_SIZE,
                                num_step=TRAIN_NUM_STEP,
                                hidden_size=HIDDEN_SIZE,
                                num_layers=NUM_LAYERS
                                )
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        valid_model = QuoraModel(
                                vocab_size=VOCAB_SIZE,
                                is_trainning=False,
                                batch_size=valid_data.len,
                                num_step=TRAIN_NUM_STEP,
                                hidden_size=HIDDEN_SIZE,
                                num_layers=NUM_LAYERS
                                )
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        predict_model = QuoraModel(
                                vocab_size=VOCAB_SIZE,
                                is_trainning=False,
                                batch_size=1,
                                num_step=TRAIN_NUM_STEP,
                                hidden_size=HIDDEN_SIZE,
                                num_layers=NUM_LAYERS
                                )
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        run_model(session, True, train_model,
                  valid_model, train_data, valid_data)


if __name__ == "__main__":
    res = main()
