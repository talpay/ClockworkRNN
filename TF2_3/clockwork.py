'''
Written for TensorFlow 2.x (tested with 2.3), supports Eager Execution.
'''

import tensorflow as tf
# import tensorflow.contrib.eager as tfe
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.python.framework import ops

from tensorflow.python.keras.layers import RNN, Layer
from tensorflow.python.keras.layers.recurrent import _config_for_enable_caching_device, DropoutRNNCellMixin, \
    _caching_device, _generate_zero_filled_state_for_cell
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.layers.base import InputSpec
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import data_structures


class CWLSTMCell(DropoutRNNCellMixin, Layer):
    def __init__(self,
                 units,
                 periods,
                 activation,
                 mask_target,
                 unidirectional=True,
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 go_backwards=False,
                 **kwargs):
        # By default use cached variable under v2 mode, see b/143699808.
        if ops.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(CWLSTMCell, self).__init__(**kwargs)

        assert units % len(periods) == 0, "Units must be divisible without remainder by number of periods (periods): units % periods == 0"

        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        assert len(periods) > 0 and units % len(periods) == 0, (
                'Unit number ({}) must be divisible '.format(units) +
                'by the number of periods since periods are equally sized')

        assert mask_target in ['c', 'h', 'ch', 'i']

        self.mask_target = mask_target
        self.unidirectional = unidirectional
        self.periods = periods

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        if self.recurrent_dropout != 0 and implementation != 1:
            logging.debug(
                'RNN `implementation=2` is not supported when `recurrent_dropout` is set. Using `implementation=1`.')
            self.implementation = 1
        else:
            self.implementation = implementation

        self.state_size = data_structures.NoDependency([self.units, self.units, self.units])
        self.output_size = self.units

        self.go_backwards = go_backwards

        n = self.units // len(self.periods)
        module_mask = np.zeros((self.units, self.units), np.float32) # W_hh shape: (h, 4*h)
        periods_z = np.zeros((self.units,), np.float32)
        for i, t in enumerate(self.periods):
            module_mask[i * n:, i * n:(i + 1) * n] = 1
            periods_z[i * n:(i + 1) * n] = t

        #module_mask = np.repeat(module_mask, 4, axis=0).transpose() # right->left
        module_mask = np.repeat(module_mask, 4, axis=-1)
        self.periods = tf.Variable(periods_z, trainable=False, name='periods')
        self.module_mask = tf.Variable(module_mask, trainable=False, name='module_mask')

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.get('ones')((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer

            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device)
        else:
            self.bias = None

        if self.unidirectional:
            self.recurrent_kernel = self.recurrent_kernel * self.module_mask

        self.built = True

    def _compute_carry_and_output(self, x_arr, h_tm1_arr, h_tm1, c_tm1, mod_ts):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x_arr
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1_arr
        h_upd = False

        i = self.recurrent_activation(
            x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
        f = self.recurrent_activation(x_f + K.dot(
            h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))

        # candidate states before SBA
        c = f * c_tm1 + i * self.activation(x_c + K.dot(
            h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
        o = self.recurrent_activation(
            x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
        h = o * self.activation(c)

        if self.mask_target == 'i':
            equal = tf.equal(mod_ts, tf.zeros_like(i))
            i = tf.where(equal, i, tf.zeros_like(i))

            c = f * c_tm1 + i * self.activation(x_c + K.dot(
                h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
            o = self.recurrent_activation(
                x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
            h_upd = True

        if self.mask_target in ['c', 'ch']:
            equal = tf.equal(mod_ts, tf.zeros_like(c))
            c = tf.where(equal, c, c_tm1)

        if self.mask_target in ['h', 'ch']:
            equal = tf.equal(mod_ts, tf.zeros_like(h))
            h = tf.where(equal, h, h_tm1)

        return c, o, h, h_upd

    def _compute_carry_and_output_fused(self, z, c_tm1):
        """Computes carry and output using fused kernels."""
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        return c, o, True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        time_step = states[2] # counter t

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)

        mod_ts = tf.math.floormod(time_step, self.periods)

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
            k_i, k_f, k_c, k_o = array_ops.split(
                self.kernel, num_or_size_splits=4, axis=1)
            x_i = K.dot(inputs_i, k_i)
            x_f = K.dot(inputs_f, k_f)
            x_c = K.dot(inputs_c, k_c)
            x_o = K.dot(inputs_o, k_o)
            if self.use_bias:
                b_i, b_f, b_c, b_o = array_ops.split(
                    self.bias, num_or_size_splits=4, axis=0)
                x_i = K.bias_add(x_i, b_i)
                x_f = K.bias_add(x_f, b_f)
                x_c = K.bias_add(x_c, b_c)
                x_o = K.bias_add(x_o, b_o)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            x_arr = (x_i, x_f, x_c, x_o)
            h_tm1_arr = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
            c, o, h, h_upd = self._compute_carry_and_output(x_arr, h_tm1_arr, h_tm1, c_tm1, mod_ts)

        else:  # impl 2 which is not supported
            if 0. < self.dropout < 1.:
                inputs = inputs * dp_mask[0]
            z = K.dot(inputs, self.kernel)
            z += K.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z = array_ops.split(z, num_or_size_splits=4, axis=1)
            c, o, h_upd = self._compute_carry_and_output_fused(z, c_tm1)

        if h_upd:
            h = o * self.activation(c)  # because of this line we can't subclass the LSTM like civilized people would...

        return h, [h, c, tf.math.add(time_step, 1)]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # inputs =
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_config(self):
        config = {
            'units':
                self.units,
            'periods': self.periods,
            'unidirectional': self.unidirectional,
            'mask_target': self.mask_target,
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'unit_forget_bias':
                self.unit_forget_bias,
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout,
            'implementation':
                self.implementation
        }
        config.update(_config_for_enable_caching_device(self))
        base_config = super(CWLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype))


class CWLSTM(RNN):
    def __init__(self,
                 units,
                 periods,
                 activation,
                 mask_target,
                 unidirectional=True,
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if implementation == 0:
            logging.warning('`implementation=0` has been deprecated, '
                            'and now defaults to `implementation=1`.'
                            'Please update your layer call.')
        if 'enable_caching_device' in kwargs:
            cell_kwargs = {'enable_caching_device':
                               kwargs.pop('enable_caching_device')}
        else:
            cell_kwargs = {}
        cell = CWLSTMCell(
            units,
            periods=periods,
            activation=activation,
            unidirectional=unidirectional,
            mask_target=mask_target,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            unit_forget_bias=unit_forget_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation,
            go_backwards=go_backwards,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True),
            **cell_kwargs)
        super(CWLSTM, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._maybe_reset_cell_dropout_mask(self.cell)
        return super(CWLSTM, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def periods(self):
        return self.cell.periods

    @property
    def decay(self):
        return self.cell.decay

    @property
    def mask_target(self):
        return self.cell.mask_target

    @property
    def unidirectional(self):
        return self.cell.unidirectional

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    def get_config(self):
        config = {
            'units':
                self.units,
            'periods':
                self.periods,
            'unidirectional':
                self.unidirectional,
            'mask_target':
                self.mask_target,
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'unit_forget_bias':
                self.unit_forget_bias,
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout,
            'implementation':
                self.implementation
        }
        config.update(_config_for_enable_caching_device(self.cell))
        base_config = super(CWLSTM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return cls(**config)


def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [
            K.in_train_phase(dropped_inputs, ones, training=training)
            for _ in range(count)
        ]
    return K.in_train_phase(dropped_inputs, ones, training=training)




class CWRNNCell(DropoutRNNCellMixin, Layer):
    def __init__(self,
                 units,
                 periods,
                 unidirectional=True,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        # By default use cached variable under v2 mode, see b/143699808.
        if ops.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(CWRNNCell, self).__init__(**kwargs)

        assert units % len(periods) == 0, "Units must be divisible without remainder by number of periods (periods): units % len(periods) == 0"

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.unidirectional = unidirectional

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = data_structures.NoDependency([self.units, self.units])
        self.output_size = self.units

        self.periods = periods

        n = self.units // len(self.periods)
        module_mask = np.zeros((self.units, self.units), np.float32)
        periods_z = np.zeros((self.units,), np.float32)
        for i, t in enumerate(self.periods):
            module_mask[i * n:, i * n:(i + 1) * n] = 1
            periods_z[i * n:(i + 1) * n] = t

        self.periods = tf.Variable(periods_z, trainable=False, name='periods')
        self.module_mask = tf.Variable(module_mask, trainable=False, name='module_mask')

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name='bias',
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device)
        else:
            self.bias = None

        if self.unidirectional:
            self.recurrent_kernel = self.recurrent_kernel * self.module_mask

        self.built = True

    def call(self, inputs, states, training=None):

        # prev_output = states[0] if nest.is_sequence(states) else states
        prev_output = states[0]
        time_step = states[1]

        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training)

        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, self.kernel)
        else:
            h = K.dot(inputs, self.kernel)
        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_output = prev_output * rec_dp_mask

        output = h + K.dot(prev_output, self.recurrent_kernel)
        if self.activation is not None:
            output = self.activation(output)

        mod_ts = tf.math.floormod(time_step, self.periods)

        # build bool vectors for activation at each timestep
        equal = tf.equal(mod_ts, tf.zeros_like(output))

        output = tf.where(equal, output, prev_output)
        # new_state = [output] if nest.is_sequence(states) else output
        new_state = [output, tf.math.add(time_step, 1)]

        return output, new_state

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # inputs =
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout,
            'periods':
                self.periods,
            'unidirectional':
                self.unidirectional
        }
        config.update(_config_for_enable_caching_device(self))
        base_config = super(CWRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CWRNN(RNN):
    def __init__(self,
                 units,
                 periods,
                 unidirectional=True,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
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
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `SimpleRNN` has been deprecated. '
                            'Please remove it from your layer call.')
        if 'enable_caching_device' in kwargs:
            cell_kwargs = {'enable_caching_device':
                               kwargs.pop('enable_caching_device')}
        else:
            cell_kwargs = {}
        cell = CWRNNCell(
            units,
            periods=periods,
            unidirectional=unidirectional,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True),
            **cell_kwargs)
        super(CWRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._maybe_reset_cell_dropout_mask(self.cell)
        return super(CWRNN, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def periods(self):
        return self.cell.periods

    @property
    def unidirectional(self):
        return self.cell.unidirectional

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {
            'units':
                self.units,
            'periods':
                self.periods,
            'unidirectional':
                self.unidirectional,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(CWRNN, self).get_config()
        config.update(_config_for_enable_caching_device(self.cell))
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)
