import tensorflow as tf
import random
from config import HOP_LENGTH, MESSAGE_POOL, WINDOW_LENGTH

def random_message():
    return random.choice(MESSAGE_POOL)

def embed(embedder, signal, message=random_message()):
    stft = tf.transpose(tf.signal.stft(signal, WINDOW_LENGTH, HOP_LENGTH, WINDOW_LENGTH), perm=[0,2,1])
    stft = tf.stack([tf.math.real(stft),tf.math.imag(stft)], axis=-1)
    output = embedder([stft, message])
    stft_complex = tf.complex(output[:,:,:,0], output[:,:,:,1])
    return tf.signal.inverse_stft(tf.transpose(stft_complex, perm=[0,2,1]), WINDOW_LENGTH, HOP_LENGTH, WINDOW_LENGTH, window_fn=tf.signal.inverse_stft_window_fn(HOP_LENGTH))

def detect(detector, signal):
    stft = tf.transpose(tf.signal.stft(signal, WINDOW_LENGTH, HOP_LENGTH, WINDOW_LENGTH), perm=[0,2,1])
    stft = tf.stack([tf.math.real(stft),tf.math.imag(stft)], axis=-1)
    return detector(stft)

