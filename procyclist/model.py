import tensorflow as tf


class ContinuousSequenceModel:
    """
    Constructs the continuous sequence prediction model

    Parameters
    ----------
    config : ContinuousSequenceModelConfig
        Configuration for model

    See Also
    --------
    DefaultCSMConfig
    """

    def __init__(self, config):
        # Input and target placeholders
        in_shape = [None, None, config.input_dim]
        out_shape = [None, None, config.output_dim]
        with tf.variable_scope('input'):
            inputs = tf.placeholder(tf.float32, in_shape, name='inputs')
            targets = tf.placeholder(tf.float32, out_shape, name='targets')
            sequence_length = tf.placeholder(tf.int32, [None])

        self._config = config
        self._build_graph(inputs, targets, sequence_length, config)

    def _build_graph(self, inputs, targets, sequence_length, config):
        output_dim = config.output_dim
        n_hidden = config.n_hidden
        batch_size = tf.shape(inputs)[0]
        n_steps = tf.shape(inputs)[1]

        with tf.variable_scope('LSTM'):
            dropout = tf.placeholder_with_default(1.0, shape=[], name='dropout')
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                n_hidden, forget_bias=1.0, dropout_keep_prob=dropout)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [cell] * config.n_layers, state_is_tuple=True)

        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs, state = tf.nn.dynamic_rnn(
            cell, inputs, initial_state=initial_state,
            sequence_length=sequence_length, swap_memory=True)
        output = tf.reshape(outputs, [-1, n_hidden])

        with tf.variable_scope('regression'):
            W1 = tf.get_variable('W1', [n_hidden, n_hidden], dtype=tf.float32)
            b1 = tf.get_variable('b1', [n_hidden], dtype=tf.float32)
            z1 = tf.nn.relu(tf.matmul(output, W1) + b1)
            z1 = tf.nn.dropout(z1, dropout)

            W2 = tf.get_variable('W2', [n_hidden, n_hidden], dtype=tf.float32)
            b2 = tf.get_variable('b2', [n_hidden], dtype=tf.float32)
            z2 = tf.nn.relu(tf.matmul(z1, W2) + b2)
            z2 = tf.nn.dropout(z2, dropout)

            W3 = tf.get_variable('W3', [n_hidden, output_dim], dtype=tf.float32)
            b3 = tf.get_variable('b3', [output_dim], dtype=tf.float32)
            prediction = tf.matmul(z2, W3) + b3

        with tf.variable_scope('loss'):
            target = tf.reshape(targets, [-1, output_dim])
            mask = tf.sequence_mask(
                sequence_length, maxlen=n_steps, dtype=tf.bool)
            mask = tf.reshape(mask, [-1, output_dim])
            deltas = tf.boolean_mask(prediction - target, mask)
            loss = tf.reduce_mean(deltas ** 2)

        with tf.variable_scope('optimization'):
            gvs = config.optimizer.compute_gradients(loss)
            grads, vals = zip(*gvs)
            clipped_grads, grad_norm = tf.clip_by_global_norm(grads, 1e3)
            clipped_gvs = list(zip(clipped_grads, vals))
            train_op = config.optimizer.apply_gradients(clipped_gvs)

        self._dropout = dropout
        self._grad_norm = grad_norm
        self._initial_state = initial_state
        self._inputs = inputs
        self._loss = loss
        self._predictions = tf.reshape(prediction, [-1, n_steps, output_dim])
        self._sequence_length = sequence_length
        self._state = state
        self._targets = targets
        self._train_op = train_op

    @property
    def dropout(self):
        return self._dropout

    @property
    def grad_norm(self):
        return self._grad_norm

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def loss(self):
        return self._loss

    @property
    def predictions(self):
        return self._predictions

    @property
    def state(self):
        return self._state

    @property
    def train_op(self):
        return self._train_op

    @property
    def config(self):
        return self._config


class SequenceModel(ContinuousSequenceModel):
    """
    Model for predicting output after final step.
    """

    def __init__(self, config):
        # Input and target placeholders
        in_shape = [None, None, config.input_dim]
        out_shape = [None, config.output_dim]
        with tf.variable_scope('input'):
            inputs = tf.placeholder(tf.float32, in_shape, name='inputs')
            targets = tf.placeholder(tf.float32, out_shape, name='targets')
            sequence_length = tf.placeholder(tf.int32, [None])

        self._config = config
        self._build_graph(inputs, targets, sequence_length, config)


    def _build_graph(self, inputs, targets, sequence_length, config):
        output_dim = config.output_dim
        n_hidden = config.n_hidden
        batch_size = tf.shape(inputs)[0]

        lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            n_hidden, forget_bias=1.0, dropout_keep_prob=0.6)
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell] * config.n_layers, state_is_tuple=True)

        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs, state = tf.nn.dynamic_rnn(
            cell, inputs, initial_state=initial_state,
            sequence_length=sequence_length, swap_memory=True)

	# Retrieve last output of LSTM
        sample_indices = tf.range(batch_size)
        last_output_indices = tf.pack([sample_indices, sequence_length], axis=1)
        output = tf.gather_nd(outputs, last_output_indices)

        with tf.variable_scope('regression'):
            W = tf.get_variable('W', [n_hidden, output_dim], dtype=tf.float32)
            b = tf.get_variable('b', [output_dim], dtype=tf.float32)
            W = tf.Print(W, [tf.shape(b)], 'W shape: ')
            b = tf.Print(b, [tf.shape(b)], 'b shape: ')
            predictions = tf.matmul(output, W) + b

        with tf.variable_scope('loss'):
            deltas = predictions - targets
            loss = tf.reduce_mean(deltas ** 2)

        with tf.variable_scope('optimization'):
            train_op = config.optimizer.minimize(loss)

        self._initial_state = initial_state
        self._inputs = inputs
        self._loss = loss
        self._predictions = predictions
        self._sequence_length = sequence_length
        self._state = state
        self._targets = targets
        self._train_op = train_op
