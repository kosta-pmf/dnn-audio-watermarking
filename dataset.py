import numpy as np
from config import BATCH_SIZE, MESSAGE_POOL, NUM_BITS, POOL_SIZE, SIGNAL_SHAPE
import tensorflow as tf
from random import randint

def create_message_pool(pool_size=1<<14, message_size=64):
    pool = set([])
    while len(pool) < pool_size:
        pool.add(tuple(np.random.randint(0,2,message_size)))

    return np.array(list(pool))

def generate_random_message(message_pool=MESSAGE_POOL, batch_size=BATCH_SIZE, num_bits=NUM_BITS):
    ind = randint(0, len(message_pool)-1)
    return np.broadcast_to(message_pool[ind], (batch_size, num_bits))
  
def expand_message(message, batch_size=BATCH_SIZE, num_bits=NUM_BITS):
  temp = np.empty((batch_size, 16, 2, num_bits))
  temp[:, :, :, :] = np.expand_dims(message, axis = (1, 2))
  return temp

def generator(iterator):
    try:
        while True:
            yield [next(iterator), expand_message(generate_random_message())]
    except (RuntimeError, StopIteration):
        return

def _parse_function3d(example, stft_shape=SIGNAL_SHAPE):
    features_description = {"stft": tf.io.FixedLenFeature(np.prod(stft_shape), tf.float32)}
    features_dict = tf.io.parse_single_example(example, features_description)
    return tf.reshape(features_dict["stft"], stft_shape)