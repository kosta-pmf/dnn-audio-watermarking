import tensorflow as tf
from tensorflow.keras.layers import Layer, Add, Conv1D
from tensorflow import constant_initializer
from utils import butter_lowpass_filter
from random import shuffle, randint
import numpy as np
from config import *

class LowpassFilter(Layer):
    def __init__(self, hop_length = HOP_LENGTH, step_size = STEP_SIZE, window_length = WINDOW_LENGTH, **kwargs):
        super(LowpassFilter, self).__init__(**kwargs)
        self.hop_length = hop_length
        self.step_size = step_size
        self.window_length = window_length

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, 512,  64, 2])

    def call(self, inputs):
        print("filtfilt")
        stfts = tf.complex(inputs[:,:,:,0], inputs[:,:,:,1])
        
        signals = tf.signal.inverse_stft(tf.transpose(stfts, perm=[0,2,1]), self.window_length, self.hop_length, self.window_length, window_fn=tf.signal.inverse_stft_window_fn(self.hop_length))
        
        batch = tf.unstack(signals, axis=0, num=64)

        filtered_signals = []

        for sample in batch:
            num = np.squeeze(sample.numpy())
            filtfilt = butter_lowpass_filter(num)
            filt_tensor = tf.convert_to_tensor(filtfilt, dtype=tf.float32)
            filtered_signals.append(tf.expand_dims(filt_tensor, 1))

        filtered_signals = tf.stack(filtered_signals, 0)

        filtered_signals = tf.squeeze(filtered_signals)

        result_stfts = tf.transpose(tf.signal.stft(filtered_signals, self.window_length, self.hop_length, self.window_length), perm=[0,2,1])

        return tf.stack([tf.math.real(result_stfts),tf.math.imag(result_stfts)], axis=-1)

class AdditiveNoise(Layer):
    def __init__(self, noise_strength=NOISE_STRENGHT, coefficient=COEFFICIENT, hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH):
        super(AdditiveNoise, self).__init__()
        self.noise_strength = noise_strength
        self.coefficient = coefficient
        self.hop_length = hop_length
        self.step_size = step_size
        self.window_length = window_length

    def call(self, inputs):
      coefficient = randint(1, 100)
      if coefficient <= self.coefficient:
        print("AdditiveNoise")
        stfts = tf.complex(inputs[:,:,:,0], inputs[:,:,:,1])
        
        signals = tf.signal.inverse_stft(tf.transpose(stfts, perm=[0,2,1]), self.window_length, self.hop_length, self.window_length, window_fn=tf.signal.inverse_stft_window_fn(self.hop_length))
        
        noise_signals = Add()([tf.random.uniform(tf.shape(signals), maxval=self.noise_strength), signals])

        result_stfts = tf.transpose(tf.signal.stft(noise_signals, self.window_length, self.hop_length, self.window_length), perm=[0,2,1])

        return tf.stack([tf.math.real(result_stfts),tf.math.imag(result_stfts)], axis=-1)
      else:
        return inputs

class CuttingSamples(Layer):
    def __init__(self, num_samples=NUM_SAMPLES, coefficient=COEFFICIENT, batch_size=BATCH_SIZE, input_dim=(33215, 1), hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH, **kwargs):
        super(CuttingSamples, self).__init__(**kwargs)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.coefficient = coefficient
        self.hop_length = hop_length
        self.step_size = step_size
        self.window_length = window_length

    def call(self, inputs):
        coefficient = randint(1, 100)
        if coefficient <= self.coefficient:
            print("CuttingSamples")
            
            stfts = tf.complex(inputs[:,:,:,0], inputs[:,:,:,1])
            
            signals = tf.signal.inverse_stft(tf.transpose(stfts, perm=[0,2,1]), self.window_length, self.hop_length, self.window_length, window_fn=tf.signal.inverse_stft_window_fn(self.hop_length))
            signals = tf.expand_dims(signals, axis=-1)  
            indices = tf.random.uniform_candidate_sampler(true_classes=tf.zeros(shape=(self.batch_size, self.input_dim[0]), dtype='int64'), num_true=self.input_dim[0], unique=True, num_sampled=self.num_samples, range_max=self.input_dim[0]).sampled_candidates
            indices = tf.tile(indices, tf.constant([self.batch_size]))
            indices = Add()([indices, tf.repeat(tf.range(0,self.batch_size*self.input_dim[0], self.input_dim[0], dtype=tf.dtypes.int64), self.num_samples)])
            
            idx = tf.scatter_nd(tf.reshape(indices, shape=(self.batch_size,self.num_samples,1)), tf.ones((self.batch_size,self.num_samples)), shape=(self.batch_size*self.input_dim[0],))
            idx = tf.reshape(idx, shape=(self.batch_size,self.input_dim[0],1))
            idx_keep = tf.where(idx==0)
            idx_remove = tf.where(idx!=0)
            values_remove = tf.tile([0.0], [tf.shape(idx_remove)[0]])
            
            values_keep = tf.gather_nd(signals, idx_keep)
            signals_remove = tf.SparseTensor(idx_remove, values_remove, tf.shape(signals, out_type=tf.dtypes.int64))
            signals_keep = tf.SparseTensor(idx_keep, values_keep, tf.shape(signals, out_type=tf.dtypes.int64))
            output = tf.add(tf.sparse.to_dense(signals_remove, default_value = 0. ), tf.sparse.to_dense(signals_keep, default_value = 0.))
            output = tf.squeeze(output, axis=-1)
            
            result_stfts = tf.transpose(tf.signal.stft(output, self.window_length, self.hop_length, self.window_length), perm=[0,2,1])
            
            return tf.stack([tf.math.real(result_stfts),tf.math.imag(result_stfts)], axis=-1)

        else:
            return inputs

class ButterworthFilter(Layer):
    def __init__(self, butterworth=BUTTERWORTH, coefficient=COEFFICIENT, hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH):
        super(ButterworthFilter, self).__init__()
        self.butterworth = butterworth
        kernel_size = len(self.butterworth)
        self.conv = Conv1D(filters=1, kernel_size=kernel_size, strides=1, padding='same',kernel_initializer=constant_initializer(self.butterworth),use_bias=False,trainable=False,activation=None)
        self.coefficient = coefficient
        self.hop_length = hop_length
        self.step_size = step_size
        self.window_length = window_length

    def call(self, inputs):
      coefficient = randint(1, 100)
      if coefficient <= self.coefficient:
        print("ButterworthFilter")
        stfts = tf.complex(inputs[:,:,:,0], inputs[:,:,:,1])
        
        signals = tf.signal.inverse_stft(tf.transpose(stfts, perm=[0,2,1]), self.window_length, self.hop_length, self.window_length, window_fn=tf.signal.inverse_stft_window_fn(self.hop_length))
        
        filtered_signals = self.conv(signals)

        filtered_signals = tf.squeeze(filtered_signals)

        result_stfts = tf.transpose(tf.signal.stft(filtered_signals, self.window_length, self.hop_length, self.window_length), perm=[0,2,1])

        return tf.stack([tf.math.real(result_stfts),tf.math.imag(result_stfts)], axis=-1)
      else:
        return inputs