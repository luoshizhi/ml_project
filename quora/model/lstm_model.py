import tensorflow as tf
import inspect
import preprocessing
import time

TRAIN_BATCH_SIZE = 64
VOCAB_SIZE = 10000
TRAIN_NUM_STEP = 35
HIDDEN_SIZE = 128
NUM_LAYERS = 3
MLP_DIMENSION = [128, 2]


NUM_EPOCH = 2
KEEP_PROB = 0.5
EMBEDDING_KEEP_PROB = 0.5
MAX_GRAD_NORM = 5
LEARNING_RATE = 1.0
EVAL_BATCH_SIZE = 1


class QuoraModel(object):
    def __init__(self, vocab_size, is_trainning, batch_size, num_step,
                 hidden_size, train_keep_prob, num_layers):
        self.vocab_size = vocab_size
        self.is_trainning = is_trainning
        self.num_step = num_step
        self.hidden_size = hidden_size
        self.train_keep_prob = train_keep_prob
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


def run_epoch(session, model, batches, keep_prob, eval_op=None, verbose=True):
    fetches = [model.loss, model.accuracy, model.predict,
               model.predict_class,  model.target_class, model.out_merge,
               model.Wx_plus_b, model.layers]
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
        step += 1

        if verbose is True and step % 100 == 0:
            end = time.time()
            print ("step:{0}...".format(step),
                   "loss: {:.4f}...".format(res[0]),
                   "accuracy:{:.4f}...".format(res[1]),
                   # "predict:{0}...".format(res[2]),
                   "predict_class{0}...".format(res[3]),
                   "target_class{0}...".format(res[4]),
                   # "out_merge{0}...".format(res[5]),
                   # "Wx_plus_b{0}...".format(res[6]),
                   # "layer{0}...".format(res[7]),
                   "{:.4f} sec/100 batches".format((end - start)),)
            start = time.time()
#            return res
    return res


def main():
    initializer = tf.random_uniform_initializer(-1.0, 1.0)
    train_data = preprocessing.TextConverter("../data/train_clean_text.csv")
    train_data.build_word_index(
                vocab_size=VOCAB_SIZE,
                save_path="../data/train.vocab"+str(VOCAB_SIZE)
                )
    train_data.tran_word_to_index()
    train_data.save("../data/train_word_to_index" + str(VOCAB_SIZE) + ".csv")
    train_data.cut_padding(cut_size=TRAIN_NUM_STEP, padding="left")
    train_data.format_inputs()

    with tf.variable_scope("model", reuse=None, initializer=initializer):
        train_model = QuoraModel(
                                vocab_size=VOCAB_SIZE,
                                is_trainning=True,
                                batch_size=TRAIN_BATCH_SIZE,
                                num_step=TRAIN_NUM_STEP,
                                hidden_size=HIDDEN_SIZE,
                                train_keep_prob=KEEP_PROB,
                                num_layers=NUM_LAYERS
                                )
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        test_model = QuoraModel(
                                vocab_size=VOCAB_SIZE,
                                is_trainning=False,
                                batch_size=TRAIN_BATCH_SIZE,
                                num_step=TRAIN_NUM_STEP,
                                hidden_size=HIDDEN_SIZE,
                                train_keep_prob=1,
                                num_layers=NUM_LAYERS
                                )
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            train_batches = train_data.batch_generator(
                            batch_size=TRAIN_BATCH_SIZE)
            print("In iteration: %d" % (i + 1))
            res = run_epoch(session,
                            train_model,
                            train_batches,
                            keep_prob=0.5,
                            eval_op=train_model.optimizer,
                            verbose=True)
    return res
#            _, train_loss, train_accuracy, _= run_epoch(session,
#                                                        test_model,
#                                                        train_batches,keep_prob=1
#                                                        None,
#                                                        verbose=True)


if __name__ == "__main__":
    res = main()
