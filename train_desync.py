import numpy as np
import tensorflow as tf
import random
from random import randint
from config import NUM_BITS, BATCH_SIZE
from dataset import create_message_pool, expand_message
from train import encoderLossFunction, decoderLossFunction, customLoss

encoder_loss_weight, decoder_loss_weight = 1.0, 3.0

message_pool = create_message_pool(1)

def set_weights(step):
    global encoder_loss_weight, decoder_loss_weight

    if step<=12000:
       encoder_loss_weight = encoder_loss_weight + 0.0
       decoder_loss_weight = decoder_loss_weight - 0.0
    if step>12000 and (step-1) % 400 == 0:
       encoder_loss_weight = encoder_loss_weight + 0.2
       decoder_loss_weight = decoder_loss_weight - 0.2
    if step>16000:
       encoder_loss_weight = 3.0
       decoder_loss_weight = 1.0

def generate_random_message():
  random_number = randint(1,100)
  if random_number <= 50:
    ind = random.randint(0, len(message_pool)-1)
    return np.broadcast_to(message_pool[ind], (BATCH_SIZE, NUM_BITS))
  else:
    random_message = np.random.randint(0,2,NUM_BITS)
    while len(np.where((message_pool==random_message).all(axis=1))[0])>0:
      random_message = np.random.randint(0,2,NUM_BITS)
    
    return np.broadcast_to(random_message, (BATCH_SIZE, NUM_BITS))

def generator(iterator):
  try:
    while True:
      yield [next(iterator), expand_message(generate_random_message())]
  except (RuntimeError, StopIteration):
    return
  
def compute_loss(model, input, step):
  e, a, d = model(input)

  inputStft, inputMessage = input
  isWatermarked = e[1]
  e_loss = tf.constant(0.0)

  outputMessage = inputMessage
  if isWatermarked is False:
    print("Not watermarked")
    outputMessage = tf.zeros(inputMessage.shape)
  elif len(np.where((message_pool==inputMessage[0,0,0]).all(axis=1))[0])==0:
    print("Random watermark")
    outputMessage = tf.zeros(inputMessage.shape)
    e_loss = encoderLossFunction(inputStft, e[0], outputMessage)
  else:
    print("Watermark from pool")
    e_loss = encoderLossFunction(inputStft, e[0], outputMessage)

  d_loss = decoderLossFunction(outputMessage, d)
  return customLoss(e_loss, d_loss, step), e_loss, d_loss, isWatermarked

def train_step(embedder, detector, model, input, optimizer, step):
  with tf.GradientTape() as tape:
    loss, e_loss, d_loss, isWatermarked = compute_loss(model, input, step)
    if isWatermarked is True:
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    else:
      d_gradients = tape.gradient(decoder_loss_weight*d_loss, detector.trainable_variables)
      optimizer.apply_gradients(zip(d_gradients, detector.trainable_variables))
	
  return loss, e_loss, d_loss

def train(decoder, model, optimizer, it):
  step = 0
  for batch in generator(it):
      loss, e_loss, d_loss = train_step(decoder, model, batch, optimizer, step)
      print("batch_num:", step, "loss:", loss.numpy(), "encoder loss: ", e_loss.numpy(), "decoder loss: ", d_loss.numpy(), "elapsed time: ")
      step+=1