import dense
import complex_initializers
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import numpy as np
import pandas as pd

def load_data_dense(vars_re = [1, 2], vars_im =[1, 2], covs = [0, 0],
                    M=10000, val=0.1, test=0.1, n_features=64):
    """Generates random complex data (z = x + iy) for experiments with the 
    fully-connected CVNN layer.
    Args:
        vars_re: Real part variance of each class.
        vars_im: Imaginary part variance of each class.
        covs: Covariance between real and imaginary parts of each class.
        M: Number of training examples.
        val: Fraction of training examples for validation set.
        test: Fraction of training examples for test set.
        n_features: Number of complex input features for the neural net. 
    Returns:
        x_train, y_train: Training set.
        x_val, y_val: Validation set.
        x_test, y_test: Test set.
    """
    val = int(M*val)  # Number of validation examples
    test = int(M*test)  # Number of test examples
    
    mu = [0, 0]
    cov_matrix_0 = [[vars_re[0], covs[0]],
                    [covs[0], vars_im[0]]]
    cov_matrix_1 = [[vars_re[1], covs[1]],
                    [covs[1], vars_im[1]]]
    
    # Generate training examples
    x_train_0 = np.random.multivariate_normal(mu, cov_matrix_0, (M//2, n_features))
    y_train_0 = np.zeros(shape=(M//2, 1))
    
    x_train_1 = np.random.multivariate_normal(mu, cov_matrix_1, (M//2, n_features))
    y_train_1 = np.ones(shape=(M//2, 1))
        
    # Stack Real and imaginary parts on the same dimension
    x_train = np.zeros(shape=(M, 2*n_features))
    x_train[:M//2, :n_features] = x_train_0[:, :, 0]
    x_train[:M//2, n_features:] = x_train_0[:, :, 1]    
    x_train[M//2:, :n_features] = x_train_1[:, :, 0]
    x_train[M//2:, n_features:] = x_train_1[:, :, 1]    

    y_train = np.concatenate((y_train_0, y_train_1), axis=0)
    
    # Add noise to input
    noise = np.random.normal(scale=0.01, size=x_train.shape)
    x_train = x_train + noise
    
    # Shuffle
    assert(len(x_train) == len(y_train))
    p = np.random.permutation(len(x_train))
    x_train = x_train[p]
    y_train = y_train[p]
        
    x_train = tf.constant(x_train)
    y_train = tf.constant(y_train)
    
    x_val = x_train[-val:]
    y_val = y_train[-val:]
    x_train = x_train[:-val]
    y_train = y_train[:-val]
    
    x_test = x_train[-test:]
    y_test = y_train[-test:]
    x_train = x_train[:-test]
    y_train = y_train[:-test]
    
    return x_train, y_train, x_val, y_val, x_test, y_test

CVNN = True
x_train, y_train, x_val, y_val, x_test, y_test = load_data_dense()

inputs = keras.layers.Input(shape=(128))
if CVNN:
    x = dense.ComplexDense(
        units=32,
        activation='relu',
        kernel_initializer=complex_initializers.ComplexInitializer
        )(inputs)
    x = dense.ComplexDense(
        units=16,
        activation='relu',
        kernel_initializer=complex_initializers.ComplexInitializer
        )(inputs)
else:
    x = keras.layers.Dense(
        units=64,
        activation='relu',
        kernel_initializer='glorot_uniform'
        )(inputs)
    x = keras.layers.Dense(
        units=16,
        activation='relu',
        kernel_initializer='glorot_uniform'
        )(inputs)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.BinaryCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.BinaryAccuracy()],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=50,
    epochs=300,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)

print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=50)
print("test loss, test acc:", results)

hist_df = pd.DataFrame(history.history)
hist_csv_file = r'complex_dense\results\history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

print("\nModel summary:")
model.summary()