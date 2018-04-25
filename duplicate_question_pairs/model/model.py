import tensorflow as tf
from duplicate_question_pairs.config.config import Config


class Model:
    def __init__(self, vocab_size):
        self.config = Config()
        self.vocab_size = vocab_size
        self.construct_model()

    def construct_model(self):
        self.add_placeholders()
        self.add_embeddings()
        self.assemble_model()
        self.add_projection()
        self.add_loss()
        self.add_train_op()

    def initialise(self, sess, embed_matrix):
        sess.run(tf.global_variables_initializer())
        sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: embed_matrix})

    def add_placeholders(self):
        self.input_placeholder_q1 = tf.placeholder(tf.int32, [self.config.batchSize, self.config.maxSeqLength])
        self.input_placeholder_q2 = tf.placeholder(tf.int32, [self.config.batchSize, self.config.maxSeqLength])
        self.label_placeholder = tf.placeholder(tf.int64, [self.config.batchSize, ], name='label')
        self.dropout_placeholder = tf.placeholder(tf.float32, (), name='dropout_keep')
        self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.config.numDimensions])

    def add_embeddings(self):
        with tf.variable_scope("embeddings"):
            self.word_embedding = tf.get_variable(shape=[self.vocab_size, self.config.numDimensions],
                                                  trainable=False, name='embeddings', dtype=tf.float32)
            self.embedding_init = self.word_embedding.assign(self.embedding_placeholder)
            data_q1 = tf.nn.embedding_lookup(self.word_embedding, self.input_placeholder_q1)
            self.input_q1 = tf.cast(data_q1, tf.float32)
            data_q2 = tf.nn.embedding_lookup(self.word_embedding, self.input_placeholder_q2)
            self.input_q2 = tf.cast(data_q2, tf.float32)

    def assemble_model(self):
        def create_lstm_multicell(name, n_layers, nstates):
            def lstm_cell(i, s):
                print('creating cell %i in %s' % (i, s))
                return tf.contrib.rnn.LSTMCell(nstates, reuse=tf.get_variable_scope().reuse)

            lstm_multi_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(i, name) for i in range(n_layers)])
            return lstm_multi_cell

        with tf.variable_scope('Inference', reuse=False):
            question1_multi_lstm = create_lstm_multicell('lstm1', self.config.lstm_layers, self.config.numDimensions)
            q1_initial_state = question1_multi_lstm.zero_state(self.config.batchSize, tf.float32)
            self.question1_outputs, self.question1_final_state = tf.nn.dynamic_rnn(question1_multi_lstm, self.input_q1,
                                                                                   initial_state=q1_initial_state)
        with tf.variable_scope('Inference', reuse=True) as scope:
            scope.reuse_variables()
            question2_multi_lstm = create_lstm_multicell('lstm2', self.config.lstm_layers, self.config.numDimensions)
            q2_initial_state = question2_multi_lstm.zero_state(self.config.batchSize, tf.float32)
            self.question2_outputs, self.question2_final_state = tf.nn.dynamic_rnn(question2_multi_lstm, self.input_q2,
                                                                                   initial_state=q2_initial_state)

    def add_projection(self):
        d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.question1_outputs, self.question2_outputs)),
                                  axis=1, keepdims=False))
        with tf.variable_scope("linear"):
            U = tf.get_variable(name="U", shape=[self.config.numDimensions, self.config.numClasses])
            B = tf.get_variable(name="B", shape=[self.config.batchSize, self.config.numClasses])
            self.scores = tf.add(tf.matmul(d, U), B, name='scores')
            self.prediction = tf.argmax(name="prediction", input=self.scores, axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.label_placeholder), "float"),
                                           name="accuracy")

    def add_loss(self):
        with tf.variable_scope("loss"):
            labels = tf.to_float(tf.one_hot(self.label_placeholder, depth=self.config.numClasses))
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=labels))

    def add_train_op(self):
        with tf.variable_scope("training"):
            self.train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)

    def run_batch(self, sess, train_batch_data, is_train=True, label_data=None):
        (q1_data, q2_data) = train_batch_data
        if is_train:
            drop_keep = 1.0
        else:
            drop_keep = self.config.drop_keep

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
