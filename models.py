from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, LeakyReLU, ReLU, Flatten, Dense, Concatenate, Conv2DTranspose
from tensorflow.keras.activations import sigmoid, relu
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from attacks import ButterworthFilter, AdditiveNoise, CuttingSamples
from utils import bwh
from layers import PyramidPooling2D
from random import randint
from attacks import Attacks

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
    return decoder, Model(inputs=[inp1, inp2], outputs=[e, a, d])

def get_optimizer(lr=2*1e-4, schedule_decay=1e-6):
    return Nadam(lr=lr, schedule_decay=schedule_decay)


'''
Desync classes
'''

class Encoder(Model):
  def __init__(self, **kwargs):
    super(Encoder, self).__init__(**kwargs)    

    self.conv_0 = Conv2D(filters=8, kernel_size=5, strides=1, padding='same')
    self.leakyrelu_0 = LeakyReLU(alpha=0.2)

    self.conv_1 = Conv2D(filters=16, kernel_size=5, strides=2, padding='same')
    self.batchnorm_1 = BatchNormalization()
    self.leakyrelu_1 = LeakyReLU(alpha=0.2)
      
    self.conv_2 = Conv2D(filters=32, kernel_size=5, strides=2, padding='same')
    self.batchnorm_2 = BatchNormalization()
    self.leakyrelu_2 = LeakyReLU(alpha=0.2)
      
    self.conv_3 = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')
    self.batchnorm_3 = BatchNormalization()
    self.leakyrelu_3 = LeakyReLU(alpha=0.2)
      
    self.conv_4 = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')
    self.batchnorm_4 = BatchNormalization()
    self.leakyrelu_4 = LeakyReLU(alpha=0.2)

    self.conv_5 = Conv2D(filters=256, kernel_size=5, strides=2, padding='same')
    self.batchnorm_5 = BatchNormalization()
    self.leakyrelu_5 = LeakyReLU(alpha=0.2)

    self.bottleneck = Concatenate()
      
    self.conv_6 = Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation=relu)

    self.conv_7 = Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same')
    self.concat_7 = Concatenate()# ([conv_7, conv_4])
    self.batchnorm_7 = BatchNormalization()
    self.relu_7 = ReLU()
    self.dropout_7 = Dropout(rate=0.5)

    self.conv_8 = Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same')
    self.concat_8 = Concatenate() 
    self.batchnorm_8 = BatchNormalization()
    self.relu_8 = ReLU()
    self.dropout_8 = Dropout(rate=0.5)

    self.conv_9 = Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding='same')
    self.concat_9 = Concatenate()
    self.batchnorm_9 = BatchNormalization()
    self.relu_9 = ReLU()
    self.dropout_9 = Dropout(rate=0.5)

    self.conv_10 = Conv2DTranspose(filters=16, kernel_size=5, strides=2, padding='same')
    self.concat_10 = Concatenate()
    self.batchnorm_10 = BatchNormalization()
    self.relu_10 = ReLU()

    self.conv_11 = Conv2DTranspose(filters=8, kernel_size=5, strides=2, padding='same')
    self.concat_11 = Concatenate()
    self.batchnorm_11 = BatchNormalization()
    self.relu_11 = ReLU()

    self.conv_12 = Conv2D(filters=2, kernel_size=5, strides=1, padding='same', activation=None)

  def call(self, inputs):
    inputSTFTs, inputMessages = inputs

    random_number = randint(1, 100)
    if random_number<=20:
      return inputSTFTs, False

    conv_0_out = self.conv_0(inputSTFTs)
    leakyrelu_0_out = self.leakyrelu_0(conv_0_out)

    conv_1_out = self.conv_1(leakyrelu_0_out)
    batchnorm_1_out = self.batchnorm_1(conv_1_out)
    leakyrelu_1_out = self.leakyrelu_1(batchnorm_1_out)

    conv_2_out = self.conv_2(leakyrelu_1_out)
    batchnorm_2_out = self.batchnorm_2(conv_2_out)
    leakyrelu_2_out = self.leakyrelu_2(batchnorm_2_out)

    conv_3_out = self.conv_3(leakyrelu_2_out)
    batchnorm_3_out = self.batchnorm_3(conv_3_out)
    leakyrelu_3_out = self.leakyrelu_3(batchnorm_3_out)

    conv_4_out = self.conv_4(leakyrelu_3_out)
    batchnorm_4_out = self.batchnorm_4(conv_4_out)
    leakyrelu_4_out = self.leakyrelu_4(batchnorm_4_out)
    
    conv_5_out = self.conv_5(leakyrelu_4_out)
    batchnorm_5_out = self.batchnorm_5(conv_5_out)
    leakyrelu_5_out = self.leakyrelu_5(batchnorm_5_out)

    bottleneck_out = self.bottleneck([leakyrelu_5_out, inputMessages])

    conv_6_out = self.conv_6(bottleneck_out)

    conv_7_out = self.conv_7(conv_6_out)
    concat_7_out = self.concat_7([conv_7_out, leakyrelu_4_out])
    batchnorm_7_out = self.batchnorm_7(concat_7_out)
    relu_7_out = self.relu_7(batchnorm_7_out)
    dropout_7_out = self.dropout_7(relu_7_out)

    conv_8_out = self.conv_8(dropout_7_out)
    concat_8_out = self.concat_8([conv_8_out, leakyrelu_3_out])
    batchnorm_8_out = self.batchnorm_8(concat_8_out)
    relu_8_out = self.relu_8(batchnorm_8_out)
    dropout_8_out = self.dropout_8(relu_8_out)

    conv_9_out = self.conv_9(dropout_8_out)
    concat_9_out = self.concat_9([conv_9_out, leakyrelu_2_out])
    batchnorm_9_out = self.batchnorm_9(concat_9_out)
    relu_9_out = self.relu_9(batchnorm_9_out)
    dropout_9_out = self.dropout_9(relu_9_out)

    conv_10_out = self.conv_10(dropout_9_out)
    concat_10_out = self.concat_10([conv_10_out, leakyrelu_1_out])
    batchnorm_10_out = self.batchnorm_10(concat_10_out)
    relu_10_out = self.relu_10(batchnorm_10_out)

    conv_11_out = self.conv_11(relu_10_out)
    concat_11_out = self.concat_11([conv_11_out, leakyrelu_0_out])
    batchnorm_11_out = self.batchnorm_11(concat_11_out)
    relu_11_out = self.relu_11(batchnorm_11_out)
    
    conv_12_out = self.conv_12(relu_11_out)
    return conv_12_out

class Decoder(Model):

  def __init__(self):
    super(Decoder, self).__init__()

    self.conv_1 = Conv2D(filters=32, kernel_size=5, strides=2, padding='same')
    self.batchnorm_1 = BatchNormalization()
    self.leakyrelu_1 = LeakyReLU(alpha=0.2)

    self.conv_2 = Conv2D(filters=32, kernel_size=5, strides=2, padding='same')
    self.batchnorm_2 = BatchNormalization()
    self.leakyrelu_2 = LeakyReLU(alpha=0.2)

    self.conv_3 = Conv2D(filters=64, kernel_size=5, strides=(1, 2), padding='same')
    self.batchnorm_3 = BatchNormalization()
    self.leakyrelu_3 = LeakyReLU(alpha=0.2)

    self.conv_4 = Conv2D(filters=64, kernel_size=5, strides=(1, 2), padding='same')
    self.batchnorm_4 = BatchNormalization()
    self.leakyrelu_4 = LeakyReLU(alpha=0.2)

    self.conv_5 = Conv2D(filters=128, kernel_size=5, strides=(1, 2), padding='same')
    self.batchnorm_5 = BatchNormalization()
    self.leakyrelu_5 = LeakyReLU(alpha=0.2)

    self.conv_6 = Conv2D(filters=128, kernel_size=5, strides=(1, 2), padding='same')
    self.batchnorm_6 = BatchNormalization()
    self.leakyrelu_6 = LeakyReLU(alpha=0.2)

    self.pyramidpool = PyramidPooling2D(bins=[(128, 1), (32, 1), (8, 1), (1, 1)], padding=True, method="MAX")

    self.flatten = Flatten()

    self.dense = Dense(units=NUM_BITS, activation=sigmoid)

    self.concat_input = Concatenate()

    self.t1 = Conv2DTranspose(filters=32, kernel_size=5, strides=(1, 2), padding='same') 
    self.t2 = Conv2D(filters=32, kernel_size=5, strides=2, padding='same') 
    
  def call(self, inputs):
    
    conv_1_out = self.conv_1(inputs)
    batchnorm_1_out = self.batchnorm_1(conv_1_out)
    leakyrelu_1_out = self.leakyrelu_1(batchnorm_1_out)
    
    conv_2_out = self.conv_2(leakyrelu_1_out)
    batchnorm_2_out = self.batchnorm_2(conv_2_out)
    leakyrelu_2_out = self.leakyrelu_2(batchnorm_2_out)

    conv_3_out = self.conv_3(leakyrelu_2_out)
    batchnorm_3_out = self.batchnorm_3(conv_3_out)
    leakyrelu_3_out = self.leakyrelu_3(batchnorm_3_out)

    conv_4_out = self.conv_4(leakyrelu_3_out)
    batchnorm_4_out = self.batchnorm_4(conv_4_out)
    leakyrelu_4_out = self.leakyrelu_4(batchnorm_4_out)
    
    conv_5_out = self.conv_5(leakyrelu_4_out)
    batchnorm_5_out = self.batchnorm_5(conv_5_out)
    leakyrelu_5_out = self.leakyrelu_5(batchnorm_5_out)

    conv_6_out = self.conv_6(leakyrelu_5_out)
    batchnorm_6_out = self.batchnorm_6(conv_6_out)
    leakyrelu_6_out = self.leakyrelu_6(batchnorm_6_out)

    pyramidpool_out = self.pyramidpool(leakyrelu_6_out)

    flatten_out = self.flatten(pyramidpool_out)

    out = self.dense(flatten_out)

    return  out

def get_desync_model():
  encoder = Encoder()
  attack = Attacks(dynamic=True)
  decoder = Decoder()
  inp1 = Input(SIGNAL_SHAPE)
  inp2 = Input(MESSAGE_SHAPE)
  e = encoder([inp1, inp2])
  a = attack(e)
  d = decoder(a)
  return Model(inputs=[inp1, inp2], outputs=[e, a, d])