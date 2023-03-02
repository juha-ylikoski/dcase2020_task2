"""
 @file   keras_model.py
 @brief  Script for keras model definition
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
# from import
import keras.models
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Reshape, Flatten, Lambda
from keras import backend as K

########################################################################
# keras model
########################################################################

# A function to compute the value of latent space
def compute_latent(x):
    mu, sigma = x
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch,dim))
    return mu + K.exp(sigma/2)*eps


def get_model(input_dim, latent_dim):
    """
    initial idea using image vae from 
    https://becominghuman.ai/using-variational-autoencoder-vae-to-generate-new-images-14328877e88d
    """
    encoder_input = Input(shape=input_dim)

    encoder_conv = Conv2D(filters=4, kernel_size=3, strides=2, 
                    padding='same', activation='relu')(encoder_input)
    encoder_conv = Conv2D(filters=8, kernel_size=3, strides=2, 
                    padding='same', activation='relu')(encoder_input)
    encoder = Flatten()(encoder_conv)

    mu = Dense(latent_dim)(encoder)
    sigma = Dense(latent_dim)(encoder)

    latent_space = Lambda(compute_latent, output_shape=(latent_dim,))([mu, sigma])

    # Take the convolution shape to be used in the decoder
    conv_shape = K.int_shape(encoder_conv)

    # Constructing decoder
    decoder_input = Input(shape=(latent_dim,))
    decoder = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(decoder_input)
    decoder = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(decoder)
    decoder_conv = Conv2DTranspose(filters=8, kernel_size=3, strides=2, 
                            padding='same', activation='relu')(decoder)
    decoder_conv = Conv2DTranspose(filters=4, kernel_size=3, strides=2, 
                            padding='same', activation='relu')(decoder)
    decoder_conv =  Conv2DTranspose(filters=1, kernel_size=3, 
                            padding='same', activation='sigmoid')(decoder_conv)

    # Actually build encoder, decoder and the entire VAE
    encoder = Model(encoder_input, latent_space)
    decoder = Model(decoder_input, decoder_conv)
    vae = Model(encoder_input, decoder(encoder(encoder_input)))

    return vae
#########################################################################


def load_model(file_path):
    return keras.models.load_model(file_path)

    
if __name__ == '__main__':
    vae = get_model((64,312,1),2)
    vae.summary()
