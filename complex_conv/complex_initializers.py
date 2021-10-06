import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import numpy as np

import aux_functions

class ComplexInitializer(keras.initializers.Initializer):
    def __init__(self, 
                 criterion='glorot', 
                 seed=None):
        self.criterion = criterion
        self.seed = np.random.randint(1, 100000) if seed is None else seed

    def __call__(self, shape, dtype=None):
       
        # The complex kernel shape must be separated into the shapes of the
        # real and imaginary parts
        if len(shape) == 2:
            kernel_shape = (shape[0]//2, shape[-1])
            concat_axis = 0
        else:
            kernel_shape = shape[:-1] + (shape[-1]//2,)
            concat_axis = -1
            
        fan_in, fan_out = aux_functions.compute_fans(kernel_shape)
        
        if self.criterion == 'glorot':
            s = 1/np.sqrt(fan_in + fan_out)
        elif self.criterion == 'he':
            s = 1/np.sqrt(fan_in)
        else:
            raise ValueError('Invalid criterion for '
                             'ComplexInitializer ' + self.criterion)

        abs_value = tfp.random.rayleigh(shape=kernel_shape, scale=s, 
                                        seed=self.seed)
        phase = tf.random.uniform(shape=kernel_shape, 
                                  minval=-np.pi, maxval=np.pi, seed=self.seed)
        real_kernel = abs_value * np.cos(phase)
        imag_kernel = abs_value * np.sin(phase)
        kernel = tf.concat([real_kernel, imag_kernel], axis=concat_axis)
    
        return kernel

    def get_config(self):
        return {
            'criterion': self.mode,
            'seed': self.seed
        }
