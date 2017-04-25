import keras.backend as K
from keras.engine.topology import Layer
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
import theano
import theano.tensor as T
import numpy as np

def genIndication(nparray, pad_val=0):
    shape = nparray.shape
    pad_shape = (shape[0],1,) + shape[2:]
    pad = np.ones(shape=pad_shape) * pad_val
    out = np.concatenate((pad, nparray), axis=1)
    out = out[:,:-1]
    return out

def addCell(container, cell_type, peak_dim, input_dim, name, config):
    if cell_type == 'SimpleRNN':
        return SimpleRNNCell(container, peak_dim, input_dim, name, config)
    elif cell_type == 'GRU':
        return GRUCell(container, peak_dim, input_dim, name, config)
    elif cell_type == 'ResGRU':
        return ResGRUCell(container, peak_dim, input_dim, name, config)
    else:
        raise NotImplementedError

def addAttention(container, type, encoder_dim, decoder_dim, name, config):
    if type == 'SimpleAttention':
        return SimpleAttentionModule(container, encoder_dim, decoder_dim, name, config)
    else:
        raise NotImplementedError

def addAttCell(container, cell_type, input_dim, name, config):
    if cell_type == 'GRU':
        return AttGRUCell(container, input_dim, name, config)
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
                                                      name=self.name+'_peak_initial',
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

        hh = self.activation(x_h + r * (K.dot(h_tm1, self.recurrent_kernel_h) + K.dot(peak, self.peak_kernel_h)))
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]
    
    def getInitialState(self, peak):
        state = K.dot(peak, self.peak_initial)
        return [state]

class ResGRUCell(object):
    """Gated Recurrent Unit - Cho et al. 2014.
    # Arguments
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
                                                      name=self.name+'_peak_initial',
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
        res_out = x + h
        return res_out, [res_out]
    
    def getInitialState(self, peak):
        state = K.dot(peak, self.peak_initial)
        return [state]




class DecoderContainer(Layer):
    """Decoder Container class
    # Properties
        max_time_steps: Integer, which indicates the max length to decode.
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
                name = self.name + '_inner_cell_%d_' % index + cell_type
                self.cells.append(addCell(self, cell_type, peak_dim, input_dim, name, item))
        else:
            config = self.decoder_config[0]
            cell_type = config.get('type')
            peak_dim = input_shapes[-1]
            input_dim = config.get('units')
            name = '0_' + cell_type
            self.cells.append(addCell(self, cell_type, peak_dim, input_dim, name, config))

    def __preprocessInputs(self, inputs):
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
        axes = [1, 0] + list(range(2, outputs.ndim))
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
        """
        wait to comp
        """
        if isinstance(input_shapes, list):
            return (input_shapes[0][0], self.max_time_steps, self.output_dim)
        else:
            return (input_shapes[0], self.max_time_steps, self.output_dim)

    def call(self, inputs):
        """
        wait to comp
        """
        if self.cell_num == len(inputs):
            _, peaks = self.__preprocessInputs(inputs)
            initial_states = self.__getInitialStates(peaks)
            outputs = self.__rnn(peaks, initial_states, None)
        else:
            _, peaks, indication = self.__preprocessInputs(inputs)
            initial_states = self.__getInitialStates(peaks)
            outputs = self.__rnn(peaks, initial_states, indication)
        return outputs

    def get_config(self):
        """
        wait to comp
        """
        config = {'max_time_steps': self.max_time_steps,
                  'decoder_config': self.decoder_config}
        base_config = super(DecoderContainer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AttGRUCell(object):
    """Gated Recurrent Unit - Cho et al. 2014.
    # Arguments
    """
    def __init__(self, container, input_dim, name, config):
        self.container = container
        self.input_dim = input_dim
        self.name = name
        self.units = config.get('units')
        assert self.units is not None, "Cell units MUST BE DEFINED!"

        self.activation = activations.get(config.get('activation', 'tanh'))
        self.recurrent_activation = activations.get(config.get('recurrent_activation', 'hard_sigmoid'))
        self.use_bias = config.get('use_bias', True)

        self.kernel_initializer = initializers.get(config.get('kernel_initializer', 'glorot_uniform'))
        self.recurrent_initializer = initializers.get(config.get('recurrent_initializer', 'orthogonal'))
        self.bias_initializer= initializers.get(config.get('bias_initializer', 'zeros'))

        self.kernel_regularizer = regularizers.get(config.get('kernel_regularizer', None))
        self.recurrent_regularizer = regularizers.get(config.get('recurrent_regularizer', None))
        self.bias_regularizer = regularizers.get(config.get('bias_regularize', None))
        self.activity_regularizer = regularizers.get(config.get('activity_regularizer', None))

        self.kernel_constraint = constraints.get(config.get('kernel_constraint', None))
        self.recurrent_constraint = constraints.get(config.get('recurrent_constraint', None))
        self.bias_constraint = constraints.get(config.get('bias_constraint', None))

        self.dropout = min(1., max(0., config.get('dropout', 0.)))
        self.recurrent_dropout = min(1., max(0., config.get('recurrent_dropout', 0.)))

        self.bottom = True if len(container.cells) == 0 else False
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

        if self.use_bias:
            self.bias = self.container.add_weight((self.units * 3,),
                                                  name=self.name+'_bias',
                                                  initializer='zero',
                                                  regularizer=self.bias_regularizer,
                                                  constraint=self.bias_constraint)
        else:
            self.bias = None
        
        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:, self.units: self.units * 2]
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None

    def step(self, x, state):
        h_tm1 = state # previous memory

        x_z = K.dot(x, self.kernel_z)
        x_r = K.dot(x, self.kernel_r)
        x_h = K.dot(x, self.kernel_h)
        if self.use_bias:
            x_z = K.bias_add(x_z, self.bias_z)
            x_r = K.bias_add(x_r, self.bias_r)
            x_h = K.bias_add(x_h, self.bias_h)
        z = self.recurrent_activation(x_z + K.dot(h_tm1, self.recurrent_kernel_z))
        r = self.recurrent_activation(x_r + K.dot(h_tm1, self.recurrent_kernel_r))

        hh = self.activation(x_h + r * (K.dot(h_tm1, self.recurrent_kernel_h)))
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]
    
    def getInitialState(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        state = K.zeros_like(inputs)    # (samples, timesteps, input_dim)
        state = K.sum(state, axis=(1,2))    # (samples,)
        state = K.expand_dims(state)    # (samples, 1)
        state = K.tile(state, [1, self.units])  # (samples, output_dim)
        return [state]

class SimpleAttentionModule(object):
    """SimpleAttentionModule
    # Arguments
    """
    def __init__(self, container, encoder_dim, decoder_dim, name, config):
        self.container = container
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.name = name
        self.units = config.get('units')
        assert self.units is not None, "Cell units MUST BE DEFINED!"

        self.kernel_initializer = initializers.get(config.get('kernel_initializer', 'glorot_uniform'))
        self.kernel_regularizer = regularizers.get(config.get('kernel_regularizer', None))
        self.kernel_constraint = constraints.get(config.get('kernel_constraint', None))
        # must be called
        self.build()

    def build(self):
        self.attend_kernel_Wx = self.container.add_weight((self.encoder_dim, self.units),
                                                        name=self.name+'_attend_Wx',
                                                        initializer=self.kernel_initializer,
                                                        regularizer=self.kernel_regularizer,
                                                        constraint=self.kernel_constraint)
        self.attend_kernel_Wy = self.container.add_weight((self.decoder_dim, self.units),
                                                        name=self.name+'_attend_Wy',
                                                        initializer=self.kernel_initializer,
                                                        regularizer=self.kernel_regularizer,
                                                        constraint=self.kernel_constraint)
        self.attend_kernel_b = self.container.add_weight((self.units,),
                                                        name=self.name+'_attend_b',
                                                        initializer=self.kernel_initializer,
                                                        regularizer=self.kernel_regularizer,
                                                        constraint=self.kernel_constraint)
        self.attend_kernel_v = self.container.add_weight((self.units,),
                                                        name=self.name+'_attend_v',
                                                        initializer=self.kernel_initializer,
                                                        regularizer=self.kernel_regularizer,
                                                        constraint=self.kernel_constraint)

    def step(self, top_encoder_out, output_tm1):
        e = K.dot(K.dot(top_encoder_out, self.attend_kernel_Wx) + K.dot(output_tm1, self.attend_kernel_Wy) + self.attend_kernel_b, self.attend_kernel_v)
        tmp_exp = K.exp(e)
        tmp_sum = K.sum(tmp_exp, axis=(0,))
        alpha = tmp_exp / tmp_sum
        context = K.batch_dot(alpha.dimshuffle([1,0]), top_encoder_out.dimshuffle([1,0,2]), axes=1)
        return context

class AttentionDecoderContainer(Layer):
    """Attention Decoder Container class
    # Properties
        max_time_steps: Integer, which indicates the max length to decode.
        decoder_config: A list, which contains the configs of each recurrent layer stacked by the order in list.
                        Each config is a dictionary.
    """
    def __init__(self, max_time_steps, decoder_config, attention_config, **kwargs):
        self.max_time_steps = max_time_steps
        self.decoder_config = decoder_config
        self.attention_config = attention_config
        self.output_dim = decoder_config[-1].get('units')
        self.cell_num = len(decoder_config)
        super(AttentionDecoderContainer, self).__init__(**kwargs)

    def __addCell(self, input_shapes):
        """
        wait to finish
        """
        self.cells = []
        if isinstance(input_shapes, list):
            encoder_dim = input_shapes[0][-1]
        else:
            encoder_dim = input_shapes[-1]
        for index, item in enumerate(self.decoder_config):
            cell_type = item.get('type')
            if index is not 0:
                input_dim = self.decoder_config[index-1].get('units')
            else:
                input_dim = item.get('units')
            name = self.name + '_inner_cell_%d_' % index + cell_type
            self.cells.append(addAttCell(self, cell_type, input_dim, name, item))

    def __preprocessInputs(self, inputs):
        """
        wait to comp
        """
        has_indi = False
        if len(inputs) == 2:
            indication = inputs[1]
            ndim = indication.ndim
            assert ndim >= 3, 'Indication should be at least 3D.'
            axes = [1, 0] + list(range(2, ndim))
            indication = indication.dimshuffle(axes)
            has_indi = True
        top_encoder_out = inputs[0]
        ndim = top_encoder_out.ndim
        assert ndim >= 3, 'top encoder output should be at least 3D.'
        axes = [1, 0] + list(range(2, ndim))
        top_encoder_out = top_encoder_out.dimshuffle(axes)
        if has_indi:
            return top_encoder_out, indication
        else:
            return top_encoder_out
    
    def __getInitialStates(self, inputs):
        """
        wait to comp
        """
        initial_states = []
        for index in range(self.cell_num):
            cell = self.cells[index]
            state = cell.getInitialState(inputs)
            initial_states.extend(state)
        return initial_states

    def __step(self, time, output_tm1, *states_tm1_and_top_encoder_out):
        """
        wait to comp
        """
        states_tm1 = states_tm1_and_top_encoder_out[:-1]
        top_encoder_out = states_tm1_and_top_encoder_out[-1]
        states = []
        bottom_cell = self.cells[0]
        state_tm1 = states_tm1[0]
        #context = self.attention_module.step(top_encoder_out, output_tm1)
        context = self.attention_module.step(top_encoder_out, state_tm1)
        combined_vec = K.dot(context, self.kernel_context) + K.dot(output_tm1, self.kernel_output)
        output, state = bottom_cell.step(combined_vec, state_tm1)
        output_hm1 = output
        states.extend(state)
        for index in range(1, self.cell_num):
            cell = self.cells[index]
            state_tm1 = states_tm1[index]
            output, state = cell.step(output_hm1, state_tm1)
            output_hm1 = output
            states.extend(state)
        return [output] + states

    def __rnn(self, top_encoder_out, initial_states, indication=None):
        """
        wait to comp
        """
        if indication is None:
            initial_output = K.zeros_like(initial_states[-1])
            initial_output = T.unbroadcast(initial_output, 1)
            if len(initial_states) > 0:
                initial_states[0] = T.unbroadcast(initial_states[0], 1)
            outputs, _ = theano.scan(self.__step,
                                     sequences=[T.arange(self.max_time_steps)],
                                     outputs_info=[initial_output] + initial_states,
                                     non_sequences=top_encoder_out,
                                     go_backwards=False)
            ### WARNING !!! YOU CAN NOT PUT '[' and ']' around 'peaks' WHEN call THEANO.SCAN ###
            # deal with Theano API inconsistency
        else:
            if len(initial_states) > 0:
                initial_states[0] = T.unbroadcast(initial_states[0], 1)
            outputs, _ = theano.scan(self.__step,
                                     sequences=[T.arange(self.max_time_steps), indication],
                                     outputs_info=[None] + initial_states,
                                     non_sequences=top_encoder_out,
                                     go_backwards=False)

        if isinstance(outputs, list):
            outputs = outputs[0]
        outputs = T.squeeze(outputs)
        axes = [1, 0] + list(range(2, outputs.ndim))
        outputs = outputs.dimshuffle(axes)
        return outputs

    def build(self, input_shapes):
        """
        Assumption
        """
        self.__addCell(input_shapes)
        if isinstance(input_shapes, list):
            encoder_dim = input_shapes[0][-1]
        else:
            encoder_dim = input_shapes[-1]
        decoder_dim = self.decoder_config[-1].get('units')
        attention_type = self.attention_config.get('type')
        self.attention_module = addAttention(self, attention_type, encoder_dim, decoder_dim, 'attention', self.attention_config)
        bottom_dim = self.decoder_config[0].get('units')
        self.kernel_context = self.add_weight((encoder_dim, bottom_dim),
                                              name=self.name+'_kernel_context',
                                              initializer='glorot_uniform',
                                              regularizer=None,
                                              constraint=None)
        self.kernel_output = self.add_weight((decoder_dim, bottom_dim),
                                             name=self.name+'_kernel_output',
                                             initializer='glorot_uniform',
                                             regularizer=None,
                                             constraint=None)

    @classmethod
    def stack(cls, container, cell_obj):
        """
        Sequentially add a cell_obj to the container
        """
        raise NotImplementedError

    def compute_output_shape(self, input_shapes):
        """
        wait to finish
        """
        if isinstance(input_shapes, list):
            return (input_shapes[0][0], self.max_time_steps, self.output_dim)
        else:
            return (input_shapes[0], self.max_time_steps, self.output_dim)

    def call(self, inputs):
        """
        wait to finish
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) == 1:
            top_encoder_out = self.__preprocessInputs(inputs)
            initial_states = self.__getInitialStates(inputs[0])
            outputs = self.__rnn(top_encoder_out, initial_states, None)
        else:
            top_encoder_out, indication = self.__preprocessInputs(inputs)
            initial_states = self.__getInitialStates(inputs[0])
            outputs = self.__rnn(top_encoder_out, initial_states, indication)
        return outputs

    def get_config(self):
        """
        wait to comp
        """
        config = {'max_time_steps': self.max_time_steps,
                  'decoder_config': self.decoder_config,
                  'attention_config': self.attention_config}
        base_config = super(AttentionDecoderContainer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

