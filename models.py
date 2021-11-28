from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, LeakyReLU, ReLU, Flatten, Dense, Concatenate, Conv2DTranspose
from tensorflow.keras.activations import sigmoid, relu
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from attacks import ButterworthFilter, AdditiveNoise, CuttingSamples
from utils import bwh

from config import *

def get_detector():

    decoderInput = Input(shape=SIGNAL_SHAPE)

    dconv_1 = Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(decoderInput)
    dconv_1 = BatchNormalization()(dconv_1)
    dconv_1 = LeakyReLU(alpha=0.2)(dconv_1)

    dconv_2 = Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(dconv_1)
    dconv_2 = BatchNormalization()(dconv_2)
    dconv_2 = LeakyReLU(alpha=0.2)(dconv_2)

    dconv_3 = Conv2D(filters=64, kernel_size=5, strides=(1, 2), padding='same')(dconv_2)
    dconv_3 = BatchNormalization()(dconv_3)
    dconv_3 = LeakyReLU(alpha=0.2)(dconv_3)

    dconv_4 = Conv2D(filters=64, kernel_size=5, strides=(1, 2), padding='same')(dconv_3)
    dconv_4 = BatchNormalization()(dconv_4)
    dconv_4 = LeakyReLU(alpha=0.2)(dconv_4)

    dconv_5 = Conv2D(filters=128, kernel_size=5, strides=(1, 2), padding='same')(dconv_4)
    dconv_5 = BatchNormalization()(dconv_5)
    dconv_5 = LeakyReLU(alpha=0.2)(dconv_5)

    dconv_6 = Conv2D(filters=128, kernel_size=5, strides=(1, 2), padding='same')(dconv_5)
    dconv_6 = BatchNormalization()(dconv_6) 
    dconv_6 = LeakyReLU(alpha=0.2)(dconv_6)

    flatten = Flatten()(dconv_6)

    dense = Dense(units=NUM_BITS, activation=sigmoid)(flatten) 
    return Model(inputs=decoderInput, outputs=dense)

def get_embedder(): 
  
    inputSTFT=Input(shape = SIGNAL_SHAPE)
    inputMessage=Input(shape = MESSAGE_SHAPE)


    conv_0 = Conv2D(filters=8, kernel_size=5, strides=1, padding='same')(inputSTFT) 
    conv_0 = LeakyReLU(alpha=0.2)(conv_0)

    conv_1 = Conv2D(filters=16, kernel_size=5, strides=2, padding='same')(conv_0) 
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = LeakyReLU(alpha=0.2)(conv_1)

    conv_2 = Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(conv_1) 
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = LeakyReLU(alpha=0.2)(conv_2)

    conv_3 = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(conv_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = LeakyReLU(alpha=0.2)(conv_3)

    conv_4 = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = LeakyReLU(alpha=0.2)(conv_4)

    conv_5 = Conv2D(filters=256, kernel_size=5, strides=2, padding='same')(conv_4) 
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = LeakyReLU(alpha=0.2)(conv_5)

    bottleneck = Concatenate()([conv_5, inputMessage])

    conv_6 = Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation=relu)(bottleneck) 

    upsampling_1 = Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same')(conv_6)
    upsampling_1 = Concatenate()([upsampling_1, conv_4])
 
    conv_7 = BatchNormalization()(upsampling_1)
    conv_7 = ReLU()(conv_7)
    conv_7 = Dropout(rate=0.5)(conv_7)

    upsampling_2 = Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same')(conv_7)
    upsampling_2 = Concatenate()([upsampling_2, conv_3])

    conv_8 = BatchNormalization()(upsampling_2)
    conv_8 = ReLU()(conv_8)
    conv_8 = Dropout(rate=0.5)(conv_8)

    upsampling_3 = Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding='same')(conv_8)
    upsampling_3 = Concatenate()([upsampling_3, conv_2])

    conv_9 = BatchNormalization()(upsampling_3)
    conv_9 = ReLU()(conv_9)
    conv_9 = Dropout(rate=0.5)(conv_9)

    upsampling_4 = Conv2DTranspose(filters=16, kernel_size=5, strides=2, padding='same')(conv_9)
    upsampling_4 = Concatenate()([upsampling_4, conv_1])

    conv_10 = BatchNormalization()(upsampling_4)
    conv_10 = ReLU()(conv_10)

    upsampling_5 = Conv2DTranspose(filters=8, kernel_size=5, strides=2, padding='same')(conv_10)
    upsampling_5 = Concatenate()([upsampling_5, conv_0])

    conv_11 = BatchNormalization()(upsampling_5)
    conv_11 = ReLU()(conv_11)


    output = Conv2D(filters=SIGNAL_SHAPE[-1], kernel_size=5, strides=1, padding='same', activation=None)(conv_11) 
    return Model(inputs = [inputSTFT, inputMessage], outputs = output)

def initialize_attacks():
    input = Input(shape=SIGNAL_SHAPE)
    output = ButterworthFilter()(input)
    output = AdditiveNoise()(output)
    output = CuttingSamples()(output)
    return Model(inputs=input, outputs=output)

def get_model():
    encoder = get_embedder()
    attack = initialize_attacks()
    decoder = get_detector()
    inp1 = Input(SIGNAL_SHAPE)
    inp2 = Input(MESSAGE_SHAPE)
    e = encoder([inp1, inp2])
    a = attack(e)
    d = decoder(a)
    return Model(inputs=[inp1, inp2], outputs=[e, a, d])

def get_optimizer(lr=2*1e-4, schedule_decay=1e-6):
    return Nadam(lr=lr, schedule_decay=schedule_decay)