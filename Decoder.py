import keras.backend as K
from keras.engine.topology import Layer
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
import theano
import theano.tensor as T
import numpy as np

def genIndication(nparray):
    shape = nparray.shape
    pad_shape = (shape[0],1,) + shape[2:]
    pad = np.zeros(shape=pad_shape)
    out = np.concatenate((pad, nparray), axis=1)
    out = out[:,:-1]
    return out

def addCell(container, cell_type, peak_dim, input_dim, name, config):
    if cell_type == 'SimpleRNN':
        return SimpleRNNCell(container, peak_dim, input_dim, name, config)
    elif cell_type == 'GRU':
        return GRUCell(container, peak_dim, input_dim, name, config)
    else:
        raise NotImplementedError


class SimpleRNNCell(object):
    """
    """
    def __init__(self, container, peak_dim, input_dim, name, config):
        self.container = container
        self.peak_dim = peak_dim
        self.input_dim = input_dim
        self.name = name
        self.units = config.get('units')
        assert self.units is not None, "Cell units MUST BE DEFINED!"
        self.activation = activations.get(config.get('activation', 'tanh'))
        self.use_bias = config.get('use_bias', True)
        self.kernel_initializer = initializers.get(config.get('kernel_initializer', 'glorot_uniform'))
        self.peak_initializer = initializers.get(config.get('peak_initializer', 'glorot_uniform'))
        self.recurrent_initializer = initializers.get(config.get('recurrent_initializer', 'orthogonal'))
        self.bias_initializer= initializers.get(config.get('bias_initializer', 'zeros'))
        self.kernel_regularizer = regularizers.get(config.get('kernel_regularizer', None))
        self.peak_regularizer = regularizers.get(config.get('peak_regularizer', None))
        self.recurrent_regularizer = regularizers.get(config.get('recurrent_regularizer', None))
        self.bias_regularizer = regularizers.get(config.get('bias_regularize', None))
        self.activity_regularizer = regularizers.get(config.get('activity_regularizer', None))
        self.kernel_constraint = constraints.get(config.get('kernel_constraint', None))
        self.peak_constraint = constraints.get(config.get('peak_constraint', None))
        self.recurrent_constraint = constraints.get(config.get('recurrent_constraint', None))
        self.bias_constraint = constraints.get(config.get('bias_constraint', None))
        self.dropout = min(1., max(0., config.get('dropout', 0.)))
        self.peak_dropout = min(1., max(0., config.get('peak_dropout', 0.)))
        self.recurrent_dropout = min(1., max(0., config.get('recurrent_dropout', 0.)))
        # must be called
        self.build()

    def build(self):
        self.kernel = self.container.add_weight((self.input_dim, self.units),
                                                name=self.name+'_kernel',
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint)
        self.peak_kernel = self.container.add_weight((self.peak_dim, self.units),
                                                     name=self.name+'_peak_kernel',
                                                     initializer=self.peak_initializer,
                                                     regularizer=self.peak_regularizer,
                                                     constraint=self.peak_constraint)
        self.recurrent_kernel = self.container.add_weight((self.units, self.units),
                                                          name=self.name+'_recurrent_kernel',
                                                          initializer=self.recurrent_initializer,
                                                          regularizer=self.recurrent_regularizer,
                                                          constraint=self.recurrent_constraint)
        if self.use_bias:
            self.bias = self.container.add_weight((self.units,),
                                                  name=self.name+'_bias',
                                                  initializer=self.bias_initializer,
                                                  regularizer=self.bias_regularizer,
                                                  constraint=self.bias_constraint)
        else:
            self.bias = None
        self.peak_initial = self.container.add_weight((self.peak_dim, self.units),
                                                       name=self.name+'_peak_initial',
                                                       initializer=self.peak_initializer,
                                                       regularizer=None,
                                                       constraint=None)
    
    def step(self, x, peak, state):
        last_output = state
        output = K.dot(peak, self.peak_kernel) + K.dot(last_output, self.recurrent_kernel)
        if x is not None:
            output += K.dot(x, self.kernel)
        if self.use_bias is not None:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output, [output]
    
    def getInitialState(self, peak):
        state = K.dot(peak, self.peak_initial)
        return [state]

class GRUCell(object):
    """Gated Recurrent Unit - Cho et al. 2014.
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    # References
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """
    def __init__(self, container, peak_dim, input_dim, name, config):
        self.container = container
        self.peak_dim = peak_dim
        self.input_dim = input_dim
        self.name = name
        self.units = config.get('units')
        assert self.units is not None, "Cell units MUST BE DEFINED!"

        self.activation = activations.get(config.get('activation', 'tanh'))
        self.recurrent_activation = activations.get(config.get('recurrent_activation', 'hard_sigmoid'))
        self.use_bias = config.get('use_bias', True)

        self.kernel_initializer = initializers.get(config.get('kernel_initializer', 'glorot_uniform'))
        self.peak_initializer = initializers.get(config.get('peak_initializer', 'glorot_uniform'))
        self.recurrent_initializer = initializers.get(config.get('recurrent_initializer', 'orthogonal'))
        self.bias_initializer= initializers.get(config.get('bias_initializer', 'zeros'))

        self.kernel_regularizer = regularizers.get(config.get('kernel_regularizer', None))
        self.peak_regularizer = regularizers.get(config.get('peak_regularizer', None))
        self.recurrent_regularizer = regularizers.get(config.get('recurrent_regularizer', None))
        self.bias_regularizer = regularizers.get(config.get('bias_regularize', None))
        self.activity_regularizer = regularizers.get(config.get('activity_regularizer', None))

        self.kernel_constraint = constraints.get(config.get('kernel_constraint', None))
        self.peak_constraint = constraints.get(config.get('peak_constraint', None))
        self.recurrent_constraint = constraints.get(config.get('recurrent_constraint', None))
        self.bias_constraint = constraints.get(config.get('bias_constraint', None))

        self.dropout = min(1., max(0., config.get('dropout', 0.)))
        self.peak_dropout = min(1., max(0., config.get('peak_dropout', 0.)))
        self.recurrent_dropout = min(1., max(0., config.get('recurrent_dropout', 0.)))
        # must be called
        self.build()

    def build(self):
        self.kernel = self.container.add_weight((self.input_dim, self.units * 3),
                                                name=self.name+'_kernel',
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint)
        self.recurrent_kernel = self.container.add_weight((self.units, self.units * 3),
                                                          name=self.name+'_recurrent_kernel',
                                                          initializer=self.recurrent_initializer,
                                                          regularizer=self.recurrent_regularizer,
                                                          constraint=self.recurrent_constraint)
        self.peak_kernel = self.container.add_weight((self.peak_dim, self.units * 3),
                                                     name=self.name+'_peak_kernel',
                                                     initializer=self.peak_initializer,
                                                     regularizer=self.peak_regularizer,
                                                     constraint=self.peak_constraint)
        if self.use_bias:
            self.bias = self.container.add_weight((self.units * 3,),
                                                  name=self.name+'_bias',
                                                  initializer='zero',
                                                  regularizer=self.bias_regularizer,
                                                  constraint=self.bias_constraint)
        else:
            self.bias = None
        
        self.peak_initial = self.container.add_weight((self.peak_dim, self.units),
                                                      name=self.name+'_peak_kernel',
                                                      initializer=self.peak_initializer,
                                                      regularizer=None,
                                                      constraint=None)
        
        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        self.peak_kernel_z = self.peak_kernel[:, :self.units]
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:, self.units: self.units * 2]
        self.peak_kernel_r = self.peak_kernel[:, self.units: self.units * 2]
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]
        self.peak_kernel_h = self.peak_kernel[:, self.units * 2:]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None

    #def step(self, inputs, states):
    def step(self, x, peak, state):
        h_tm1 = state # previous memory

        x_z = K.dot(x, self.kernel_z)
        x_r = K.dot(x, self.kernel_r)
        x_h = K.dot(x, self.kernel_h)
        if self.use_bias:
            x_z = K.bias_add(x_z, self.bias_z)
            x_r = K.bias_add(x_r, self.bias_r)
            x_h = K.bias_add(x_h, self.bias_h)
        z = self.recurrent_activation(x_z + K.dot(h_tm1, self.recurrent_kernel_z) + K.dot(peak, self.peak_kernel_z))
        r = self.recurrent_activation(x_r + K.dot(h_tm1, self.recurrent_kernel_r) + K.dot(peak, self.peak_kernel_r))

        #hh = self.activation(x_h + K.dot(r * h_tm1, self.recurrent_kernel_h))
        hh = self.activation(x_h + r * (K.dot(h_tm1, self.recurrent_kernel_h) + K.dot(peak, self.peak_kernel_h)))
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]
    
    def getInitialState(self, peak):
        state = K.dot(peak, self.peak_initial)
        return [state]

class DecoderContainer(Layer):
    """Decoder Container class
    # Properties
        decoder_config: A list, which contains the configs of each recurrent layer stacked by the order in list.
                        Each config is a dictionary.
    """
    def __init__(self, max_time_steps, decoder_config, **kwargs):
        self.max_time_steps = max_time_steps
        self.decoder_config = decoder_config
        self.output_dim = decoder_config[-1].get('units')
        self.cell_num = len(decoder_config)
        super(DecoderContainer, self).__init__(**kwargs)
    
    def __addCell(self, input_shapes):
        self.cells = []
        if isinstance(input_shapes, list):
            for index, item in enumerate(self.decoder_config):
                cell_type = item.get('type')
                peak_dim = input_shapes[index][-1]
                if index is not 0:
                    input_dim = self.decoder_config[index-1].get('units')
                else:
                    input_dim = self.decoder_config[-1].get('units')
                name = '%d_' % index + cell_type
                self.cells.append(addCell(self, cell_type, peak_dim, input_dim, name, item))
        else:
            config = self.decoder_config[0]
            cell_type = config.get('type')
            peak_dim = input_shapes[-1]
            input_dim = config.get('units')
            name = '0_' + cell_type
            self.cells.append(addCell(self, cell_type, peak_dim, input_dim, name, config))

    def __preprocessInputs(self, inputs):
        print("preprocess inputs Input tensors: %d." % len(inputs))
        has_indi = False
        if self.cell_num == len(inputs) - 1:
            indication = inputs[-1]
            ndim = indication.ndim
            assert ndim >= 3, 'Indication should be at least 3D.'
            axes = [1, 0] + list(range(2, ndim))
            indication = indication.dimshuffle(axes)
            inputs = inputs[:-1]
            has_indi = True
        top_encoder_out = inputs[-1]
        ndim = top_encoder_out.ndim
        assert ndim >= 3, 'top encoder output should be at least 3D.'
        axes = [1, 0] + list(range(2, ndim))
        top_encoder_out = top_encoder_out.dimshuffle(axes)
        peaks = []
        for item in inputs:
            ndim = item.ndim
            assert ndim >= 3, 'Input should be at least 3D.'
            axes = [1, 0] + list(range(2, ndim))
            item = item.dimshuffle(axes)
            print("peak ndim : %d" % item[-1].ndim)
            peaks.append(item[-1])
        if has_indi:
            return top_encoder_out, peaks, indication
        else:
            return top_encoder_out, peaks
    
    def __getInitialStates(self, peaks):
        initial_states = []
        for index in range(self.cell_num):
            cell = self.cells[index]
            peak = peaks[index]
            state = cell.getInitialState(peak)
            print("states : %d" % len(state))
            print("initial state ndim : %d" % state[0].ndim)
            initial_states.extend(state)
        return initial_states

    def __step(self, time, output_tm1, *states_tm1_and_peaks):
        states_tm1 = states_tm1_and_peaks[:self.cell_num]
        peaks = states_tm1_and_peaks[self.cell_num:]
        states = []
        bottom_cell = self.cells[0]
        peak = peaks[0]
        state_tm1 = states_tm1[0]
        output, state = bottom_cell.step(output_tm1, peak, state_tm1)
        output_hm1 = output
        states.extend(state)
        for index in range(1, self.cell_num):
            cell = self.cells[index]
            peak = peaks[index]
            state_tm1 = states_tm1[index]
            output, state = cell.step(output_hm1, peak, state_tm1)
            output_hm1 = output
            states.extend(state)
        return [output] + states
    
    def __rnn(self, peaks, initial_states, indication=None):
        if indication is None:
            initial_output = K.zeros_like(initial_states[-1])
            initial_output = T.unbroadcast(initial_output, 1)
            if len(initial_states) > 0:
                initial_states[0] = T.unbroadcast(initial_states[0], 1)
            print("before call scan peak ndim : %d" % peaks[0].ndim)
            outputs, _ = theano.scan(self.__step,
                                     sequences=[T.arange(self.max_time_steps)],
                                     outputs_info=[initial_output] + initial_states,
                                     non_sequences=peaks,
                                     go_backwards=False)
            ### WARNING !!! YOU CAN NOT PUT '[' and ']' around 'peaks' WHEN call THEANO.SCAN ###
            # deal with Theano API inconsistency
        else:
            if len(initial_states) > 0:
                initial_states[0] = T.unbroadcast(initial_states[0], 1)
            outputs, _ = theano.scan(self.__step,
                                     sequences=[T.arange(self.max_time_steps), indication],
                                     outputs_info=[None] + initial_states,
                                     non_sequences=peaks,
                                     go_backwards=False)

        if isinstance(outputs, list):
            outputs = outputs[0]
        outputs = T.squeeze(outputs)
        axes = [1,0] + list(range(2, outputs.ndim))
        outputs = outputs.dimshuffle(axes)
        return outputs

    def build(self, input_shapes):
        """
        Assumption
            
        """
        self.__addCell(input_shapes)
    
    @classmethod
    def stack(cls, container, cell_obj):
        """
        Sequentially add a cell_obj to the container
        """
        raise NotImplementedError

    def compute_output_shape(self, input_shapes):
        if isinstance(input_shapes, list):
            return (input_shapes[0][0], self.max_time_steps, self.output_dim)
        else:
            return (input_shapes[0], self.max_time_steps, self.output_dim)
   
    def call(self, inputs):
        if self.cell_num == len(inputs):
            top_encoder_out, peaks = self.__preprocessInputs(inputs)
            print("call top_encoder_out ndim : %d" % top_encoder_out.ndim)
            print("call len peaks : %d" % len(peaks))
            print("call peak ndim : %d" % peaks[0].ndim)
            initial_states = self.__getInitialStates(peaks)
            print("after __getInit... call peak ndim : %d" % peaks[0].ndim)
            print("initial states : %d" % len(initial_states))
            print("initial state ndim : %d" % (initial_states[0].ndim))
            outputs = self.__rnn(peaks, initial_states, None)
        else:
            top_encoder_out, peaks, indication = self.__preprocessInputs(inputs)
            initial_states = self.__getInitialStates(peaks)
            outputs = self.__rnn(peaks, initial_states, indication)
        return outputs