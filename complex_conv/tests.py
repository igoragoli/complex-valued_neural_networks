import tensorflow as tf
from tensorflow import keras

from conv import ComplexConv

def test_outputs_conv2d():
    layer = ComplexConv(
        rank=2,
        filters=2,
        kernel_size=3,
        strides=(1, 1),
        padding='same',
    )
    # 64 x 64 images with 4 channels each
    input_shape = tf.TensorShape((None, 64, 64, 4))
    # As the padding is 'same' and we have 2 filters, the output
    # Should be 2 channels of 64 x 64 images
    true = tf.TensorShape((None, 64, 64, 4))
    calc = layer.compute_output_shape(input_shape)
    print(calc)
    assert true.as_list() == calc.as_list()

def test_forward_conv2d():
    inputs = keras.layers.Input(shape=(128, 128, 14))
    outputs = ComplexConv(
        rank=2,
        filters=4,
        kernel_size=3,
        strides=2,
        padding='same')(inputs)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    true = (None, 64, 64, 8)
    calc = model.output_shape
    print(calc)
    assert true == calc

def test_conv2d_data():
    # Load data
    (train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()

    # Normalize
    train_x, test_x = train_x/255.0, test_x/255.0

    # Stack (Real and Imag)
    train_x = tf.concat([train_x, train_x], axis=-1)
    test_x = tf.concat([test_x, test_x], axis=-1)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    inputs = keras.layers.Input(shape=(32, 32, 6))
    x = ComplexConv(2, 32, 3, (1, 1), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = ComplexConv(2, 64, 3, (1, 1), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = ComplexConv(2, 64, 3, (1, 1), activation='relu')(x)
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)
    outputs = keras.layers.Dense(10)(x)
    
    model = keras.models.Model(inputs=inputs, outputs=outputs) 
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        train_x, train_y, 
        epochs=10, validation_data=(test_x, test_y)
    )