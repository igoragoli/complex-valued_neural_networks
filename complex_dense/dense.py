import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations, initializers, regularizers, constraints
import numpy as np

import complex_initializers

class ComplexDense(keras.layers.Layer):
    def __init__(self, 
                 units, 
                 activation, 
                 use_bias=True,
                 kernel_initializer=complex_initializers.ComplexInitializer, 
                 bias_initializer='zeros',
                 kernel_regularizer=None, 
                 bias_regularizer=None,
                 activity_regularizer=None, 
                 kernel_constraint=None,
                 bias_constraint=None, 
                 **kwargs):
        super(ComplexDense, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs)
        
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = keras.layers.InputSpec(ndim=2)
        self.supports_masking = True
        
    def build(self, input_shape):
        if input_shape[-1] % 2 != 0:
            raise Exception('The number of inputs to ComplexDense '
                            'should be even (real and imaginary parts).')
        self.kernel = self.add_weight(
            'complex_kernel',
            # The real kernel is in the first input_shape[-1]/2 lines, and 
            # the imaginary kernel is in the last input_shape[-1]/2 lines
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                'complex_bias',
                # The real bias is in the first line, and the imaginary bias is
                # in the second line
                shape=(2, self.units), 
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True
            )
        else:
            self.bias = None 
        self.built = True
    
    def call(self, inputs):
        # Separates inputs, kernel, bias in their real and imaginary parts
        input_dim = inputs.shape[-1] // 2
        real_input = inputs[:, :input_dim]
        imag_input = inputs[:, input_dim:]
        real_kernel = self.kernel[:input_dim, :]
        imag_kernel = self.kernel[input_dim:, :]
        real_bias = self.bias[0, :]
        imag_bias = self.bias[1, :]
        
        real_output = tf.matmul(real_input, real_kernel) \
            - tf.matmul(imag_input, imag_kernel)
        imag_output = tf.matmul(imag_input, real_kernel) \
            + tf.matmul(real_input, imag_kernel)
        
        if self.use_bias:
            real_output = tf.nn.bias_add(real_output, real_bias - imag_bias)
            imag_output = tf.nn.bias_add(imag_output, real_bias + imag_bias)
        
        outputs = tf.concat([real_output, imag_output], axis=-1)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs
    
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        return input_shape[:-1].concatenate(2*self.units)
        
    def get_config(self):
        config = super(ComplexDense, self).get_config()
        config.update({
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        })
        return config