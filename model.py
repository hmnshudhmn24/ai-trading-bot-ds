import tensorflow as tf
from tensorflow.keras import layers

def create_model(input_shape, action_space):
    model = tf.keras.Sequential([
        layers.Dense(64, input_shape=(input_shape,), activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(action_space, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model