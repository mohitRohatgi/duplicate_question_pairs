import tensorflow as tf
from duplicate_question_pairs.config.config import Config


class Model:
    def __init__(self, embeddings):
        self.config = Config()
        self.embeddings = embeddings
        self.construct_model()

    def construct_model(self):
        with tf.Graph().as_default():
            self.add_placeholders()
            self.add_embeddings()
            self.assemble_model()
            self.add_projection()
            self.add_loss()
            self.add_train_op()

    def add_placeholders(self):
        self.input_placeholder_q1 = tf.placeholder(tf.int32, [self.config.batchSize, self.config.maxSeqLength])
        self.input_placeholder_q2 = tf.placeholder(tf.int32, [self.config.batchSize, self.config.maxSeqLength])
        self.label_placeholder = tf.placeholder(tf.int32, [self.config.batchSize, 1], name='label')
        self.dropout_placeholder = tf.placeholder(tf.float32, (), name='dropout_keep')

    def add_embeddings(self):
        with tf.variable_scope("embeddings"):
            data_q1 = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder_q1)
            self.input_q1 = tf.cast(data_q1, tf.float32)
            data_q2 = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder_q2)
            self.input_q2 = tf.cast(data_q2, tf.float32)

    def assemble_model(self):
        def create_lstm_multicell(name, n_layers, nstates):
            def lstm_cell(i, s):
                print('creating cell %i in %s' % (i, s))
                return tf.contrib.rnn.LSTMCell(nstates, reuse=tf.get_variable_scope().reuse)

            lstm_multi_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(i, name) for i in range(n_layers)])
            return lstm_multi_cell

        with tf.variable_scope('Inference', reuse=False):
            question1_multi_lstm = create_lstm_multicell('lstm1', self.config.lstm_layers, self.config.lstmUnits)
            q1_initial_state = question1_multi_lstm.zero_state(self.config.batchSize, tf.float32)
            self.question1_outputs, self.question1_final_state = tf.nn.dynamic_rnn(question1_multi_lstm, self.input_q1,
                                                                                   initial_state=q1_initial_state)
        with tf.variable_scope('Inference', reuse=True) as scope:
            scope.reuse_variables()
            question2_multi_lstm = create_lstm_multicell('lstm2', self.config.lstm_layers, self.config.lstmUnits)
            q2_initial_state = question2_multi_lstm.zero_state(self.config.batchSize, tf.float32)
            self.question2_outputs, self.question2_final_state = tf.nn.dynamic_rnn(question2_multi_lstm, self.input_q2,
                                                                                   initial_state=q2_initial_state)

    def add_projection(self):
        d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.question1_outputs, self.question2_outputs)),
                          axis=1, keep_dims=False))
        with tf.variable_scope("linear"):
            U = tf.get_variable(name="U", shape=[self.config.numDimensions, self.config.numClasses])
            B = tf.get_variable(name="B", shape=[self.config.batchSize, ])
            scores = tf.add(tf.matmul(d, U), B)
            self.prediction = tf.argmax(name="prediction", input=scores)
            self.accuracy = tf.reduce_mean(tf.equal(self.prediction, self.label_placeholder), name="accuracy")

    def add_loss(self):
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction,
                                                                                  labels=self.label_placeholder))

    def add_train_op(self):
        self.train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)

    def run_epoch(self, sess, q1_data, q2_data, is_train=True, label_data=None):
        if is_train:
            drop_keep = 1.0
        else:
            drop_keep = 1.0

        if label_data is None:
            feed_dict = {
                self.input_placeholder_q1: q1_data,
                self.input_placeholder_q2: q2_data,
                self.dropout_placeholder: drop_keep
            }
            return sess.run([self.prediction, feed_dict])

        feed_dict = {
            self.input_placeholder_q1: q1_data,
            self.input_placeholder_q2: q2_data,
            self.label_placeholder: label_data,
            self.dropout_placeholder: drop_keep
        }

        if is_train:
            loss, accuracy, prediction, _ = sess.run([self.loss, self.accuracy, self.prediction, self.train_op],
                                                     feed_dict)
        else:
            loss, accuracy, prediction = sess.run([self.loss, self.accuracy, self.prediction], feed_dict)
        return loss, accuracy, prediction
