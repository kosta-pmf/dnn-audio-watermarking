
import tensorflow as tf
from tensorflow.keras.layers import Add, Multiply
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError
from config import BATCH_SIZE, NUM_BITS
from dataset import generator
import time
import numpy as np

encoder_loss_weight, decoder_loss_weight = 1.0, 3.0

def set_weights(step):
    global encoder_loss_weight, decoder_loss_weight

    if step>0 and step % 1400 == 0:
        encoder_loss_weight = encoder_loss_weight + 0.2
        decoder_loss_weight = decoder_loss_weight - 0.2
    if step>=14000: 
        encoder_loss_weight = 2.5
        decoder_loss_weight = 0.5

def customLoss(encoderLoss, decoderLoss, step):
    set_weights(step)
    return Add()([Multiply()([encoder_loss_weight, encoderLoss]), Multiply()([decoder_loss_weight, decoderLoss])])

def encoderLossFunction(y_true, y_pred, inputMessage):
    return MeanAbsoluteError()(y_true, y_pred)

def decoderLossFunction(y_true, y_pred):
    y_augmented = tf.squeeze(tf.slice(y_true, [0, 0, 0, 0], [BATCH_SIZE, 1, 1, NUM_BITS]))
    return BinaryCrossentropy()(y_augmented, y_pred)

def compute_loss(model, input, step):
    e, a, d = model(input)

    inputSpec, inputMessage = input
    e_loss = encoderLossFunction(inputSpec, e, inputMessage)
    d_loss = decoderLossFunction(inputMessage, d)
    return customLoss(e_loss, d_loss, step), e_loss, d_loss

def train_step(model, input, optimizer, step):
    with tf.GradientTape() as tape:
        loss, e_loss, d_loss = compute_loss(model, input, step)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, e_loss, d_loss

def train(model, optimizer, it):
    step = 0
    for batch in generator(it):
        loss, e_loss, d_loss = train_step(model, batch, optimizer, step)
        print("batch_num:", step, "loss:", loss.numpy(), "encoder loss: ", e_loss.numpy(), "decoder loss: ", d_loss.numpy())
        step += 1




