import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255.
x_test = x_test / 255.

# Encoder
encoder_input = keras.layers.Input(shape=(28, 28, 1))
x = keras.layers.Flatten()(encoder_input)
encoder_output = keras.layers.Dense(64, activation='relu')(x)

encoder = keras.Model(encoder_input, encoder_output)

# Decoder
decoder_input = keras.layers.Dense(784, activation='relu')(encoder_output)
decoder_output = keras.layers.Reshape((28, 28, 1))(decoder_input)

autoencoder = keras.Model(encoder_input, decoder_output)

autoencoder.compile('adam', 'mse')

autoencoder.fit(x_train, x_train, epochs=3, batch_size=32)

pred = autoencoder.predict([x_test[0].reshape(-1, 28, 28, 1)])[0]

# Generated image
plt.imshow(pred, cmap='gray')
plt.show()