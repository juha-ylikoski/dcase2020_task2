
import keras.models
from keras.models import Model
from keras import layers
from keras import backend as K


def get_model(input_dim, latent_dim):
    """
    initial idea using image ae from 
    https://blog.keras.io/building-autoencoders-in-keras.html
    """
    encoder_input = layers.Input(shape=input_dim)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D(2, padding='same')(x)

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu',padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    decoded = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(encoder_input, decoded)
    return autoencoder

def load_model(file_path):
    return keras.models.load_model(file_path)

    
if __name__ == '__main__':
    vae = get_model((64,64,1),2)
    vae.summary()
