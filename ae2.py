
import keras.models
from keras.models import Model
from keras import layers

def get_model(input_dim, latent_dim):
    """
    initial idea using image ae from 
    https://keras.io/examples/vision/autoencoder/
    """
    encoder_input = layers.Input(shape=input_dim)
    x = layers.Conv2D(32, 3, activation="relu",strides=2, padding="same")(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu",strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu",strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation="relu",strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Decoder
    x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(1, 3, strides=2, activation="sigmoid", padding="same")(x)
    x = layers.BatchNormalization()(x)

    autoencoder = keras.Model(encoder_input, x)
    return autoencoder

def load_model(file_path):
    return keras.models.load_model(file_path)

    
if __name__ == '__main__':
    vae = get_model((64,64,1),2)
    vae.summary()
