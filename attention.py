# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras import backend as K
from keras import activations, initializations, regularizers
from keras.engine import Layer, InputSpec


def time_distributed_dense(x, w, b=None, dropout=None,
                           input_dim=None, output_dim=None, timesteps=None):
    '''Apply y.w + b for every temporal slice y of x.
    '''
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b:
        x = x + b
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x


class FlexibleRecurrent(Layer):
    '''Abstract base class for recurrent layers.
    Do not use in a model -- it's not a valid layer!
    Use its children classes `LSTM`, `GRU` and `SimpleRNN` instead.

    All recurrent layers (`LSTM`, `GRU`, `SimpleRNN`) also
    follow the specifications of this class and accept
    the keyword arguments listed below.

    # Example

    ```python
        # as the first layer in a Sequential model
        model = Sequential()
        model.add(LSTM(32, input_shape=(10, 64)))
        # now model.output_shape == (None, 32)
        # note: `None` is the batch dimension.

        # the following is identical:
        model = Sequential()
        model.add(LSTM(32, input_dim=64, input_length=10))

        # for subsequent layers, not need to specify the input size:
        model.add(LSTM(16))
    ```

    # Arguments
        weights: list of Numpy arrays to set as initial weights.
            The list should have 3 elements, of shapes:
            `[(input_dim, output_dim), (output_dim, output_dim), (output_dim,)]`.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False). If True, the network will be unrolled,
            else a symbolic loop will be used. When using TensorFlow, the network
            is always unrolled, so this argument does not do anything.
            Unrolling can speed-up a RNN, although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        consume_less: one of "cpu", "mem", or "gpu" (LSTM/GRU only).
            If set to "cpu", the RNN will use
            an implementation that uses fewer, larger matrix products,
            thus running faster on CPU but consuming more memory.
            If set to "mem", the RNN will use more matrix products,
            but smaller ones, thus running slower (may actually be faster on GPU)
            while consuming less memory.
            If set to "gpu" (LSTM/GRU only), the RNN will combine the input gate,
            the forget gate and the output gate into a single matrix,
            enabling more time-efficient parallelization on the GPU. Note: RNN
            dropout must be shared for all gates, resulting in a slightly
            reduced regularization.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        input_length: Length of input sequences, to be specified
            when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
            Note that if the recurrent layer is not the first layer
            in your model, you would need to specify the input length
            at the level of the first layer
            (e.g. via the `input_shape` argument)

    # Input shape
        3D tensor with shape `(nb_samples, timesteps, input_dim)`.

    # Output shape
        - if `return_sequences`: 3D tensor with shape
            `(nb_samples, timesteps, output_dim)`.
        - else, 2D tensor with shape `(nb_samples, output_dim)`.

    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.

    # Note on performance
        You are likely to see better performance with RNNs in Theano compared
        to TensorFlow. Additionally, when using TensorFlow, it is often
        preferable to set `unroll=True` for better performance.

    # Note on using statefulness in RNNs
        You can set RNN layers to be 'stateful', which means that the states
        computed for the samples in one batch will be reused as initial states
        for the samples in the next batch.
        This assumes a one-to-one mapping between
        samples in different successive batches.

        To enable statefulness:
            - specify `stateful=True` in the layer constructor.
            - specify a fixed batch size for your model, by passing
                if sequential model:
                  a `batch_input_shape=(...)` to the first layer in your model.
                else for functional model with 1 or more Input layers:
                  a `batch_shape=(...)` to all the first layers in your model.
                This is the expected shape of your inputs *including the batch size*.
                It should be a tuple of integers, e.g. `(32, 10, 100)`.

        To reset the states of your model, call `.reset_states()` on either
        a specific layer, or on your entire model.
    '''
    def __init__(self, weights=None,
                 return_sequences=False, go_backwards=False, stateful=False,
                 unroll=False, consume_less='cpu',
                 input_dim=None, input_length=None, **kwargs):
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.consume_less = consume_less

        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(Recurrent, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.output_dim)
        else:
            return (input_shape[0], self.output_dim)

    def compute_mask(self, input, mask):
        if self.return_sequences:
            return mask
        else:
            return None

    def step(self, x, states):
        raise NotImplementedError

    def get_constants(self, x):
        return []

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        #initial_state = K.tile(initial_state, [1, self.output_dim])  # (samples, output_dim)
        #initial_states = [initial_state for _ in range(len(self.states))]
        initial_states = []
        for i in range(len(self.states)):
            this_initial_state = K.tile(initial_state, [1, self.dim_list[i]])
            initial_states.append(this_initial_state)
        return initial_states

    def preprocess_input(self, x):
        return x

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = K.int_shape(x)
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, x)

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def get_config(self):
        config = {'return_sequences': self.return_sequences,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll,
                  'consume_less': self.consume_less}
        if self.stateful and self.input_spec[0].shape:
            config['batch_input_shape'] = self.input_spec[0].shape
        else:
            config['input_dim'] = self.input_dim
            config['input_length'] = self.input_length

        base_config = super(Recurrent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AttentionSimpleRNN(FlexibleRecurrent):
    '''Fully-connected RNN where the output is to be fed back to input.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        encoder_list: representation list got from encoder
        encoder_shape: tuple - (input_time_steps, encoder_feature_dim)
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim, h_dim, s_dim, attd_dim,
                 encoder_list, encoder_shape,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        # new added
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.attd_dim = attd_dim

        self.encoder_list = encoder_list
        self.encoder_shape = encoder_shape

        self.dim_list = [self.output_dim, self.h_dim, self.encoder_shape[1], self.encoder_shape[0], self.s_dim]

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(AttentionSimpleRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None,None,None,None,None]
        input_dim = input_shape[2]
        #self.input_dim = input_dim

        ###### attention impl ######

        self.by = self.add_weight((self.output_dim,),
                                    initializer=self.init,
                                    name='{}_by'.format(self.name),
                                    regularizer=None)

        self.Why = self.add_weight((self.h_dim, self.output_dim),
                                    initializer=self.init,
                                    name='{}_Why'.format(self.name),
                                    regularizer=None)

        self.Wch = self.add_weight((self.encoder_shape[1], self.h_dim),
                                    initializer=self.init,
                                    name='{}_Wch'.format(self.name),
                                    regularizer=None)

        # for cal e and w

        self.v = self.add_weight((self.attd_dim,),
                                  initializer=self.init,
                                  name='{}_v'.format(self.name),
                                  regularizer=None)

        self.W = self.add_weight((self.s_dim, self.attd_dim),
                                  initializer=self.init,
                                  name='{}_W'.format(self.name),
                                  regularizer=None)

        self.V = self.add_weight((self.encoder_shape[1], self.attd_dim),
                                  initializer=self.init,
                                  name='{}_V'.format(self.name),
                                  regularizer=None)

        self.b = self.add_weight((self.attd_dim,),
                                  initializer=self.init,
                                  name='{}_b'.format(self.name),
                                  regularizer=None)

        # for cal e and w
        

        self.Wys = self.add_weight((self.output_dim, self.s_dim),
                                    initializer=self.init,
                                    name='{}_Wys'.format(self.name),
                                    regularizer=None)
        
        self.Whh = self.add_weight((self.h_dim, self.h_dim),
                                    initializer=self.init,
                                    name='{}_Whh'.format(self.name),
                                    regularizer=None)

        self.Wcs = self.add_weight((self.encoder_shape[1], self.s_dim),
                                    initializer=self.init,
                                    name='{}_Wcs'.format(self.name),
                                    regularizer=None)

        self.Wss = self.add_weight((self.s_dim, self.s_dim),
                                    initializer=self.init,
                                    name='{}_Wss'.format(self.name),
                                    regularizer=None)

        ###### attention impl ######

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')
        if hasattr(self, 'states'):
            K.set_value(self.states[0], np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1], np.zeros((input_shape[0], self.h_dim)))
            K.set_value(self.states[2], np.zeros((input_shape[0], self.encoder_shape[1])))
            K.set_value(self.states[3], np.zeros((input_shape[0], self.encoder_shape[0])))
            K.set_value(self.states[4], np.zeros((input_shape[0], self.s_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.h_dim)),
                           K.zeros((input_shape[0], self.encoder_shape[1])),
                           K.zeros((input_shape[0], self.encoder_shape[0])),
                           K.zeros((input_shape[0], self.s_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = K.int_shape(x)
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            return time_distributed_dense(x, self.W, self.b, self.dropout_W,
                                          input_dim, self.output_dim,
                                          timesteps)
        else:
            return x

    def step(self, x, states):

        ###### attention impl ######

        prev_y = states[0]
        prev_h = states[1]
        prev_c = states[2]
        prev_e = states[3]
        prev_s = states[4]
        B_Y = states[5]
        B_H = states[6]
        B_C = states[7]
        B_S = states[8]

        this_s = self.activation(K.dot(prev_s * B_S, self.Wss) + K.dot(prev_c * B_C, self.Wcs) + K.dot(prev_y * B_Y, self.Wys))

        tmp = K.dot(this_s,self.W) + self.b
        tmp = K.repeat(tmp,self.encoder_shape[0])
        this_e = K.dot(self.activation(tmp + K.dot(self.encoder_list,self.V)), self.v)
        tmp_w = K.exp(this_e)
        sum_w = K.sum(tmp_w,axis=1,keepdims=True)
        this_w = tmp_w / sum_w

        this_c = K.batch_dot(this_w, self.encoder_list, axes=(1,1))

        this_h = self.activation(K.dot(this_c * B_C, self.Wch) + K.dot(prev_h * B_H, self.Whh))
        #this_h = self.activation(K.dot(this_c, self.Wch) + K.dot(prev_h, self.Whh) + K.dot(this_s, self.Wsh))
        this_y = self.activation(K.dot(this_h * B_H, self.Why))

        return prev_y, [this_y,this_h,this_c,this_e,this_s]

        ###### attention impl ######

    def get_constants(self, x):
        constants = []
        constants.append(K.cast_to_floatx(1.))
        constants.append(K.cast_to_floatx(1.))
        constants.append(K.cast_to_floatx(1.))
        constants.append(K.cast_to_floatx(1.))
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

