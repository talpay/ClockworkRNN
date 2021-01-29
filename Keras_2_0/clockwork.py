'''
Written for legacy code and providing compatibility with Keras Standalone 2.0.8
(before it fully migrated to tf.keras)
'''

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from keras.layers.recurrent import *

def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
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
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x

def normalize_tuple(value, n, name):
    """Transforms a single int or iterable of ints into an int tuple.

    # Arguments
        value: The value to validate and convert. Could an int, or any iterable
          of ints.
        n: The size of the tuple to be returned.
        name: The name of the argument being validated, e.g. "strides" or
          "kernel_size". This is only used to format error messages.

    # Returns
        A tuple of n integers.

    # Raises
        ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        if len(value_tuple) != n:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except ValueError:
                raise ValueError('The `' + name + '` argument must be a tuple of ' +
                                 str(n) + ' integers. Received: ' + str(value) + ' '
                                 'including element ' + str(single_value) + ' of type' +
                                 ' ' + str(type(single_value)))
    return value_tuple

class CWRNN(Recurrent):
    """
    # Arguments
        units: Positive integer, dimensionality of the output space.
        periods: Periods that define the periodic activations.
        unidirectional: Has nothing to do with bidirectional rnn!
            Disable unidirectional (right->left) connection scheme,
            i.e. you are left with an unmasked normal recurrent matrix.
            This leads to bad results and is only useful to validate
            the original connection scheme.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.

    # References
        - Clockwork RNN: https://arxiv.org/abs/1402.3511
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 periods=[1],
                 unidirectional=True,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(CWRNN, self).__init__(**kwargs)

        if self.implementation != 0:
            assert K.backend() != 'theano', ('Implementation modes other than 0 might only work with tensorflow')

        self.units = units
        self.periods = np.asarray(periods)

        assert len(periods) > 0 and units % len(periods) == 0, (
            'Unit number ({}) must be divisible '.format(units) +
            'by the number of periods ({}) since modules are equally sized '.format(len(periods)) +
            'and each module must have its own period.')

        mod_size = units // len(periods)
        print('mod-size = ', mod_size)

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.unidirectional = unidirectional

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_spec = [InputSpec(shape=(None, self.units)),
                           InputSpec(shape=(None, self.units))]

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))

        # track previous state and time step (for periodic activation)
        self.states = [None, None] # first 2 entries of self.states
        if self.stateful:
            self.reset_states()

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        # binary period mask for recurrent_kernel (set left-right connections to 0)
        n = self.units // len(self.periods)
        module_mask = np.zeros((self.units, self.units), K.floatx())
        periods = np.zeros((self.units,), np.int16)
        for i, t in enumerate(self.periods):
            module_mask[i * n:, i * n:(i + 1) * n] = 1
            periods[i * n:(i + 1) * n] = t

        module_mask = K.variable(module_mask, name='module_mask')
        self.periods = K.variable(periods, name='periods')

        if self.unidirectional:
            self.recurrent_kernel *= module_mask

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def preprocess_input(self, inputs, training=None):
        if self.implementation > 0:
            return inputs
        else: # impl 0 activates in below function instead of step
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            return _time_distributed_dense(inputs,
                                           self.kernel,
                                           self.bias,
                                           self.dropout,
                                           input_dim,
                                           self.units,
                                           timesteps,
                                           training=training)

    def step(self, inputs, states):
        # contents of states-list:
        #   0 - h_tm1
        #   1 - t
        #   2 - dropout mask
        #   3 - recurrent dropout mask
        prev_output, time_step, do, rdo = states

        if self.implementation == 0:
            h = inputs
        else:
            if 0 < self.dropout < 1:
                h = K.dot(inputs * states[2], self.kernel)
            else:
                h = K.dot(inputs, self.kernel)

            if self.bias is not None:
                h = K.bias_add(h, self.bias)

        # In this implementation, dropout is overshadowed by the binary clocking
        if 0 < self.recurrent_dropout < 1:
            prev_output *= rdo

        output = h + K.dot(prev_output, self.recurrent_kernel)
        if self.activation is not None:
            output = self.activation(output)

        # clocking: decision on which units get activated
        if K.backend() == 'tensorflow':
            # this "hack" is sadly necessary for tensorflow: K.switch uses tf.cond but we need tf.where
            import tensorflow as tf
            output = tf.where(tf.equal(tf.mod(time_step, self.periods), K.zeros_like(output)), output, prev_output)
            # output = tf.where(K.equal(time_step % self.periods, 0.), output, prev_output)
        else:
            output = K.switch(K.equal(time_step % self.periods, 0.), output, prev_output)

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            output._uses_learning_phase = True

        return output, [output, time_step + 1]


    def get_initial_states(self, x):
        initial_states = super(CWRNN, self).get_initial_states(x)
        # in a bidirectional net, we intentionally reverse the clocking:
        # this makes sense conceptionally for the vanilla model but it really depends on what you're trying to do.
        if self.go_backwards:
            print('Warning: Bidirectional CWRNN was tested but might be buggy.')
            input_length = self.input_spec[0].shape[1]
            initial_states[1] = float(input_length)
        else:
            initial_states[1] = K.variable(0.)
        return initial_states

    def reset_states(self, states=None):
        print('Warning: Stateful CWRNN has not been tested thoroughly.')
        # TODO test and add super-call or necessary parts thereof
        if self.go_backwards:
            initial_time = self.input_spec[0].shape[1]
        else:
            initial_time = 0.

        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((self.input_spec[0].shape[0], self.units)))
            K.set_value(self.states[1], initial_time)
        else:
            self.states = [K.zeros((self.input_spec[0].shape[0], self.units)),
                           K.variable(initial_time)]



    def get_constants(self, inputs, training=None):
        constants = []
        if self.implementation != 0 and 0 < self.dropout < 1:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = K.in_train_phase(dropped_inputs,
                                       ones,
                                       training=training)
            constants.append(dp_mask)
        else:
            constants.append(K.cast_to_floatx(1.))

        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)
            rec_dp_mask = K.in_train_phase(dropped_inputs,
                                           ones,
                                           training=training)
            constants.append(rec_dp_mask)
        else:
            constants.append(K.cast_to_floatx(1.))
        return constants

    def get_config(self):
        config = {'units': self.units,
                  'periods': self.periods,
                  'unidirectional': self.unidirectional,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(CWRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
