import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import activations, initializers, regularizers, constraints
import numpy as np
import functools

import complex_initializers

class ComplexConv(keras.layers.Layer):
    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=1,
                 groups=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=complex_initializers.ComplexInitializer,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 conv_op=None,
                 **kwargs):
        super(ComplexConv, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs
        )
        self.rank = rank
        self.filters = filters
        self.groups = groups  # NOTE: Maybe remove
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = keras.layers.InputSpec(min_ndim=self.rank + 2)
        
        self._is_causal = self.padding == 'causal'
        self._channels_first = self.data_format == 'channels_first'
        self._tf_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)
        
    def build(self, input_shape):
        if input_shape[-1] % 2 != 0:
            raise Exception('The number of input feature maps to ComplexConv '
                            'should be even (real and imaginary parts).')

        input_channel = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        
        # The complex kernels are separated in their real and imaginary parts.
        # Therefore we divide input_channel by 2, and we make kernel_shape[-1] 
        # to be equal to 2*self.filters.
        kernel_shape = self.kernel_size + (input_channel//self.groups//2,
                                           2*self.filters)
        
        self.kernel = self.add_weight(
            'complex_kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.bias_constraint,
            trainable=True,
            dtype=self.dtype
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='complex_bias',
                shape=(2*self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = keras.layers.InputSpec(min_ndim=self.rank + 2,
            axes={channel_axis: input_channel})
        
        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__
        if tf_op_name == 'Conv1D':
            tf_op_name = 'conv1d'  # Backwards compat.
        
        self._convolution_op = functools.partial(
            tf.nn.convolution,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name
        )
        self.built = True
    
    def call(self, inputs):
        # Separates inputs, kernel, bias in their real and imaginary parts
        input_dim = inputs.shape[-1] // 2
        real_inputs = inputs[..., :input_dim]
        imag_inputs = inputs[..., input_dim:]
        real_kernel = self.kernel[..., :self.filters]
        imag_kernel = self.kernel[..., self.filters:]
        real_bias = self.bias[:self.filters]
        imag_bias = self.bias[self.filters:]
        
        real_outputs = self._convolution_op(real_inputs, real_kernel) \
            - self._convolution_op(imag_inputs, imag_kernel)
        imag_outputs = self._convolution_op(real_inputs, imag_kernel) \
            + self._convolution_op(imag_inputs, real_kernel)
        
        if self.use_bias:
            real_outputs = tf.nn.bias_add(real_outputs, real_bias - imag_bias)
            imag_outputs= tf.nn.bias_add(imag_outputs, real_bias + imag_bias)
            
        outputs = tf.concat([real_outputs, imag_outputs], axis=-1)
        
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
    
    def _spatial_output_shape(self, spatial_input_shape):
        return [
            conv_utils.conv_output_length(
                length,
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            for i, length in enumerate(spatial_input_shape)
        ]
    
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        return tf.TensorShape(
            input_shape[:batch_rank]
            + self._spatial_output_shape(input_shape[batch_rank:-1])
            + [2*self.filters]
        )
    
    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'groups': self.groups,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return -1 - self.rank
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                                'should be defined. Found `None`.')
        return int(input_shape[channel_axis])
