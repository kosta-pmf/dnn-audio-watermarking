import tensorflow as tf
from tensorflow.keras.layers import Layer, Add, Conv1D, ZeroPadding1D, Multiply
from tensorflow import constant_initializer
from utils import butter_lowpass_filter
from random import randint, sample
import numpy as np
from config import *


class BaseAttack(Layer):
    def __init__(self, probability=15, hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH, message="BaseAttack", **kwargs):
        super(BaseAttack, self).__init__(**kwargs)
        self.probability = probability
        self.hop_length = hop_length
        self.step_size = step_size
        self.window_length = window_length
        self.message = message

class FrequencyFilter(Layer):
  def __init__(self, sampling_rate=16000, cutoff_frequency=2000, stft_shape=SIGNAL_SHAPE, **kwargs):
    super(FrequencyFilter, self).__init__(**kwargs)
    self.sampling_rate = sampling_rate
    self.cutoff_frequency = cutoff_frequency
    num_ones = int(stft_shape[0]*(cutoff_frequency/sampling_rate))
    zeros = tf.zeros((BATCH_SIZE, stft_shape[0]-num_ones, stft_shape[1], 2), dtype="float32")
    ones = tf.ones((BATCH_SIZE, num_ones, stft_shape[1], 2), dtype="float32")
    self.mask = tf.concat([ones, zeros], axis=1)

  def call(self, inputs):
    return Multiply()([inputs, self.mask])

class LowpassFilter(BaseAttack):
    def __init__(self, hop_length = HOP_LENGTH, step_size = STEP_SIZE, window_length = WINDOW_LENGTH, message="filtfilt", **kwargs):
        super().__init__(self, hop_length, step_size, window_length, message, **kwargs)

    def call(self, inputs):
        print(self.message)
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
    
    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, 512,  64, 2])

class AdditiveNoise(BaseAttack):
    def __init__(self, noise_strength=NOISE_STRENGHT, probability=COEFFICIENT, hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH, message="AdditiveNoise", **kwargs):
        super().__init__(self, probability, hop_length, step_size, window_length, message, **kwargs)
        self.noise_strength = noise_strength
        self.probability = probability
        self.hop_length = hop_length
        self.step_size = step_size
        self.window_length = window_length

    def call(self, inputs):
      random_number = randint(1, 100)
      if random_number <= self.probability:
        print(self.message)
        stfts = tf.complex(inputs[:,:,:,0], inputs[:,:,:,1])
        
        signals = tf.signal.inverse_stft(tf.transpose(stfts, perm=[0,2,1]), self.window_length, self.hop_length, self.window_length, window_fn=tf.signal.inverse_stft_window_fn(self.hop_length))
        
        noise_signals = Add()([tf.random.uniform(tf.shape(signals), maxval=self.noise_strength), signals])

        result_stfts = tf.transpose(tf.signal.stft(noise_signals, self.window_length, self.hop_length, self.window_length), perm=[0,2,1])

        return tf.stack([tf.math.real(result_stfts),tf.math.imag(result_stfts)], axis=-1)
      else:
        return inputs

class CuttingSamples(BaseAttack):
    def __init__(self, num_samples=NUM_SAMPLES, probability=COEFFICIENT, batch_size=BATCH_SIZE, input_dim=(33215, 1), hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH, message="AdditiveNoise", **kwargs):
        super().__init__(self, probability, hop_length, step_size, window_length, message, **kwargs)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.input_dim = input_dim

    def call(self, inputs):
        random_number = randint(1, 100)
        if random_number <= self.probability:
            print(self.message)
            
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

class ButterworthFilter(BaseAttack):
  def __init__(self, butterworth=[], probability=15, hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH, message="ButterworthFilter", **kwargs):
    super().__init__(self, probability, hop_length, step_size, window_length, message, **kwargs)
    self.butterworth = butterworth
    kernel_size = len(self.butterworth)
    self.conv = Conv1D(filters=1, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer=constant_initializer(self.butterworth),use_bias=False,trainable=False,activation=None)

  def call(self, inputs):
    random_number = randint(1, 100)
    if random_number <= self.probability:
      print(self.message)
      stfts = tf.complex(inputs[:,:,:,0], inputs[:,:,:,1])
      signals = tf.signal.inverse_stft(tf.transpose(stfts, perm=[0,2,1]), self.window_length, self.hop_length, self.window_length, window_fn=tf.signal.inverse_stft_window_fn(self.hop_length))
      filtered_signals = self.conv(tf.expand_dims(signals, axis=-1))
      filtered_signals = tf.squeeze(filtered_signals)
      result_stfts = tf.transpose(tf.signal.stft(filtered_signals, self.window_length, self.hop_length, self.window_length), perm=[0,2,1])
      
      return tf.stack([tf.math.real(result_stfts),tf.math.imag(result_stfts)], axis=-1)
    else:
      return inputs

  def compute_output_shape(self, input_shape):
    return tf.TensorShape([None, 512,  64, 2])

class FlipSamples(BaseAttack):
  def __init__(self, samples_to_flip=1000, batch_size=BATCH_SIZE, probability=15, hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH, message="FlipSamples", **kwargs):
    super().__init__(self, probability, hop_length, step_size, window_length, message, **kwargs)
    self.samples_to_flip = samples_to_flip
    self.batch_size = batch_size

  def call(self, inputs):
    random_number = randint(1, 100)
    if random_number <= self.probability:
      stfts = tf.complex(inputs[:,:,:,0], inputs[:,:,:,1])
      signals = tf.signal.inverse_stft(tf.transpose(stfts, perm=[0,2,1]), self.window_length, self.hop_length, self.window_length, window_fn=tf.signal.inverse_stft_window_fn(self.hop_length))
      signals = tf.expand_dims(signals, axis=-1)

      signal_length = tf.shape(signals)[1]
      
      to_flip = randint(max(2, self.samples_to_flip-1000), self.samples_to_flip)
      print(self.message," to flip:", to_flip)
      
      indices = tf.random.uniform_candidate_sampler(true_classes=tf.zeros(shape=(self.batch_size, signal_length), dtype='int64'), num_true=signal_length, unique=True, num_sampled=to_flip, range_max=signal_length).sampled_candidates
      indices = tf.tile(indices, tf.constant([self.batch_size]))
      indices = Add()([indices, tf.repeat(tf.range(0,self.batch_size*signal_length, signal_length, dtype=tf.dtypes.int64), to_flip)])
      
      idx = tf.scatter_nd(tf.reshape(indices, shape=(self.batch_size,to_flip,1)), tf.ones((self.batch_size,to_flip)), shape=(self.batch_size*signal_length,))
      idx = tf.reshape(idx, shape=(self.batch_size, signal_length,1))
      idx_keep = tf.where(idx==0)
      idx_flip = tf.where(idx!=0)
      
      values_keep = tf.gather_nd(signals, idx_keep)
      values_flip = tf.gather_nd(signals, tf.random.shuffle(idx_flip))
      
      signals_flip = tf.SparseTensor(idx_flip, values_flip, tf.shape(signals, out_type=tf.dtypes.int64))
      signals_keep = tf.SparseTensor(idx_keep, values_keep, tf.shape(signals, out_type=tf.dtypes.int64))
      outputs = tf.add(tf.sparse.to_dense(signals_flip, default_value = 0. ), tf.sparse.to_dense(signals_keep, default_value = 0.))

      outputs = tf.squeeze(outputs, axis=-1)
      result_stfts = tf.transpose(tf.signal.stft(outputs, self.window_length, self.hop_length, self.window_length), perm=[0,2,1])
      return tf.stack([tf.math.real(result_stfts),tf.math.imag(result_stfts)], axis=-1)
      
    return inputs

  def compute_output_shape(self, input_shape):
    return tf.TensorShape([None, 512,  64, 2])

class SampleCutting(BaseAttack):
    def __init__(self, samples_to_cut=1000, signal_length=33216, probability=15, hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH, message="SampleCutting", **kwargs):
        super().__init__(self, probability, hop_length, step_size, window_length, message, **kwargs)
        self.samples_to_cut = samples_to_cut
        self.signal_length = signal_length

    def call(self, inputs):
      random_number = randint(1, 100)
      if random_number <= self.probability:
        print(self.message)
        stfts = tf.complex(inputs[:,:,:,0], inputs[:,:,:,1])
        signals = tf.signal.inverse_stft(tf.transpose(stfts, perm=[0,2,1]), self.window_length, self.hop_length, self.window_length, window_fn=tf.signal.inverse_stft_window_fn(self.hop_length))
        
        input_list = tf.unstack(signals, axis=1)
        to_cut = randint(max(0, self.samples_to_cut-1000), self.samples_to_cut)
        print(self.message,"to cut:", to_cut)
        indices = sample(range(0, self.signal_length), to_cut)
        for index in sorted(indices, reverse = True):
          del input_list[index]

        outputs = tf.stack(input_list, 1)

        result_stfts = tf.transpose(tf.signal.stft(outputs, self.window_length, self.hop_length, self.window_length), perm=[0,2,1])
        return tf.stack([tf.math.real(result_stfts),tf.math.imag(result_stfts)], axis=-1)
      
      return inputs

    def compute_output_shape(self, input_shape):
      temp = tf.random.normal((1, self.signal_length - self.samples_to_cut, ))
      stft = tf.signal.stft(temp, self.window_length, self.hop_length, self.window_length)
      return tf.TensorShape([None, 512,  stft.shape[1], 2])

class Delay(BaseAttack):
  def __init__(self, delay=4000, factor=0.2, signal_length=33216, probability=15, hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH, message="Echo", **kwargs):
    super().__init__(self, probability, hop_length, step_size, window_length, message, **kwargs)
    self.delay = delay
    self.factor = factor
    self.signal_length = signal_length

  def call(self, inputs):
    random_number = randint(1, 100)
    if random_number <= self.probability:
      to_delay = randint(max(0, self.delay-4000), self.delay)
      print(self.message, "to delay:", to_delay)
      stfts = tf.complex(inputs[:,:,:,0], inputs[:,:,:,1])
      signals = tf.signal.inverse_stft(tf.transpose(stfts, perm=[0,2,1]), self.window_length, self.hop_length, self.window_length, window_fn=tf.signal.inverse_stft_window_fn(self.hop_length))
      signals = tf.expand_dims(signals, axis=-1)
      
      delays = ZeroPadding1D((to_delay, 0))(signals)
      padded_signals = ZeroPadding1D((0, to_delay))(signals)
      outputs = Add()([padded_signals, Multiply()([delays, tf.broadcast_to(self.factor, shape=tf.shape(delays))])])

      outputs = tf.squeeze(outputs, axis=-1)
      result_stfts = tf.transpose(tf.signal.stft(outputs, self.window_length, self.hop_length, self.window_length), perm=[0,2,1])
      return tf.stack([tf.math.real(result_stfts),tf.math.imag(result_stfts)], axis=-1)
    
    return inputs
    
  def compute_output_shape(self, input_shape):
    temp = tf.random.normal((1, self.signal_length + self.delay, ))
    stft = tf.signal.stft(temp, self.window_length, self.hop_length, self.window_length)
    return tf.TensorShape([None, 512,  stft.shape[1], 2])

class Downsampling(BaseAttack):
  def __init__(self, factor=2, signal_length=33216, probability=15, hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH, message="Downsampling", **kwargs):
    super().__init__(self, probability, hop_length, step_size, window_length, message, **kwargs)
    self.factor = factor
    self.signal_length = signal_length

    self.keep_indices = range(0, self.signal_length, self.factor)
    self.delete_indices = list(set(range(0, self.signal_length)).difference(self.keep_indices))
    self.delete_indices.reverse()

  def call(self, inputs):
    random_number = randint(1, 100)
    if random_number <= self.probability:
      print(self.message)
      stfts = tf.complex(inputs[:,:,:,0], inputs[:,:,:,1])
      signals = tf.signal.inverse_stft(tf.transpose(stfts, perm=[0,2,1]), self.window_length, self.hop_length, self.window_length, window_fn=tf.signal.inverse_stft_window_fn(self.hop_length))
      signals_list = tf.unstack(signals, axis=1)

      for ind in self.delete_indices:
        del signals_list[ind]

      outputs = tf.stack(signals_list, axis=1)

      result_stfts = tf.transpose(tf.signal.stft(outputs, self.window_length, self.hop_length, self.window_length), perm=[0,2,1])
      return tf.stack([tf.math.real(result_stfts),tf.math.imag(result_stfts)], axis=-1)
    
    return inputs

  def compute_output_shape(self, input_shape):
    temp = tf.random.normal((1, self.keep_indices, ))
    stft = tf.signal.stft(temp, self.window_length, self.hop_length, self.window_length)
    return tf.TensorShape([None, 512,  stft.shape[1], 2])

class Upsampling(BaseAttack):
  def __init__(self, factor=2, signal_length=33216, probability=15, hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH, message="Upsampling", **kwargs):
    super().__init__(self, probability, hop_length, step_size, window_length, message, **kwargs)
    self.factor = factor
    self.signal_length = signal_length
    self.interpolation_filter = np.bartlett(2*self.factor+1)
    self.conv = Conv1D(filters=1, kernel_size=len(self.interpolation_filter), padding="same", strides=1,kernel_initializer=constant_initializer(self.interpolation_filter),use_bias=False,trainable=False,activation=None)

  def call(self, inputs):
    random_number = randint(1, 100)
    if random_number <= self.probability:
      print(self.message)
      stfts = tf.complex(inputs[:,:,:,0], inputs[:,:,:,1])
      signals = tf.signal.inverse_stft(tf.transpose(stfts, perm=[0,2,1]), self.window_length, self.hop_length, self.window_length, window_fn=tf.signal.inverse_stft_window_fn(self.hop_length))
      
      ffts = tf.signal.fft(tf.cast(signals, dtype='complex64'))
      expanded_ffts = tf.concat([ffts for i in range(self.factor)], axis=1)
      expanded_signals = tf.math.real(tf.signal.ifft(expanded_ffts))

      expanded_signals_list = tf.unstack(expanded_signals, axis=1)
      for ind in range(1, self.factor):
        expanded_signals_list.pop()

      expanded_signals = tf.stack(expanded_signals_list, axis=1)
      expanded_signals = tf.expand_dims(expanded_signals, axis=-1)

      outputs = self.conv(expanded_signals)

      outputs = tf.squeeze(outputs, axis=-1)
      result_stfts = tf.transpose(tf.signal.stft(outputs, self.window_length, self.hop_length, self.window_length), perm=[0,2,1])
      return tf.stack([tf.math.real(result_stfts),tf.math.imag(result_stfts)], axis=-1)
    
    return inputs

  def compute_output_shape(self, input_shape):
    temp = tf.random.normal((1, self.signal_length*self.factor-(self.factor-1), ))
    stft = tf.signal.stft(temp, self.window_length, self.hop_length, self.window_length)
    return tf.TensorShape([None, 512,  stft.shape[1], 2])


class TimeFold(BaseAttack):
  def __init__(self, batch_size=BATCH_SIZE, signal_length=33216, segment_length=1000, overlap_length=50, hop_size=1100, probability=15, hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH, message="TimeFold", **kwargs):
    super().__init__(self, probability, hop_length, step_size, window_length, message, **kwargs)
    self.batch_size = batch_size
    self.signal_length = signal_length
    self.segment_length = segment_length
    self.overlap_length = overlap_length
    self.hop_size = hop_size-self.segment_length

  def prepare(self):

    if self.hop_size < 500:
      to_hop = randint(1, self.hop_size)
    else:
      to_hop = randint(self.hop_size-300, self.hop_size)
    
    print(self.message,"to hop:", to_hop)

    t = self.signal_length//(self.segment_length+to_hop)
    if self.signal_length % (self.segment_length+to_hop) >= self.segment_length:
      t+=1
    self.output_length = self.segment_length+(t-1)*(self.segment_length-self.overlap_length)

    self.delete_indices = []
    cnt = 0
    segment_end = False
    while cnt<self.signal_length:
      if segment_end is True:
        self.delete_indices.extend(range(cnt, min(cnt+to_hop, self.signal_length)))
        cnt += to_hop
        segment_end = False
      else:
        if cnt+self.segment_length > self.signal_length:
          self.delete_indices.extend(range(cnt, self.signal_length))
          
        cnt += self.segment_length
        segment_end = True
    
    self.delete_indices.reverse()

    self.final_delete_indices = []
    cnt = self.segment_length
    i = 0
    while i < t-1:
      self.final_delete_indices.extend(range(cnt, cnt+self.overlap_length))
      cnt += self.segment_length
      i += 1

    self.final_delete_indices.reverse()

    self.coeff_inc = tf.pad(tf.tile(tf.pad(tf.range(1,self.overlap_length+1), tf.constant([0, self.segment_length-self.overlap_length], shape=(1,2))), tf.constant(t-1, shape=(1,))), tf.constant([self.segment_length-self.overlap_length, self.overlap_length], shape=(1,2)))
    self.coeff_inc = tf.cast(self.coeff_inc, tf.float32)/self.overlap_length
    self.coeff_inc = tf.reshape(tf.tile(self.coeff_inc, tf.constant(self.batch_size, shape=(1,))), (self.batch_size,-1, 1))

    self.coeff_dec = tf.pad(tf.tile(tf.pad(tf.range(self.overlap_length, limit=0, delta=-1), tf.constant([self.segment_length-self.overlap_length, 0], shape=(1,2))), tf.constant(t-1, shape=(1,))), tf.constant([0, self.segment_length], shape=(1,2)))
    self.coeff_dec = tf.cast(self.coeff_dec, tf.float32)/self.overlap_length
    self.coeff_dec = tf.reshape(tf.tile(self.coeff_dec, tf.constant(self.batch_size, shape=(1,))), (self.batch_size,-1, 1))
    
    self.coeff_const = tf.tile(tf.pad(tf.ones(shape=(self.segment_length-2*self.overlap_length)), tf.constant([self.overlap_length, self.overlap_length], shape=(1,2))), tf.constant(t-2, shape=(1,)))
    self.coeff_const = tf.pad(tf.pad(self.coeff_const, tf.constant([self.overlap_length ,0], shape=(1,2))), tf.constant([self.segment_length-self.overlap_length, 0], shape=(1,2)), constant_values=1)
    self.coeff_const = tf.pad(tf.pad(self.coeff_const, tf.constant([0, self.overlap_length], shape=(1,2))), tf.constant([0, self.segment_length-self.overlap_length], shape=(1,2)), constant_values=1)
    self.coeff_const = tf.cast(self.coeff_const, tf.float32)
    self.coeff_const = tf.reshape(tf.tile(self.coeff_const, tf.constant(self.batch_size, shape=(1,))), (self.batch_size,-1, 1))

  def call(self, inputs):
    random_number = randint(1, 100)
    if random_number <= self.probability:
      self.prepare()
      stfts = tf.complex(inputs[:,:,:,0], inputs[:,:,:,1])
      signals = tf.signal.inverse_stft(tf.transpose(stfts, perm=[0,2,1]), self.window_length, self.hop_length, self.window_length, window_fn=tf.signal.inverse_stft_window_fn(self.hop_length))
      signals = tf.expand_dims(signals, axis=-1)

      signal_length = signals.shape[1]

      input_list = tf.unstack(signals, axis=1)
      for ind in self.delete_indices:
        del input_list[ind]
      signals = tf.stack(input_list, axis=1)

      for ind in range(self.overlap_length-1,-1,-1):
        del input_list[ind]

      shifted_signals = tf.pad(tf.stack(input_list, axis=1),  tf.constant([[0,0], [0, self.overlap_length], [0,0]], shape=(3,2)))
      out1 = Multiply()([self.coeff_const, signals])
      out2 = Multiply()([self.coeff_dec, signals])
      out3 = Multiply()([self.coeff_inc, shifted_signals])

      outputs = Add()([out1, out2, out3])

      output_list = tf.unstack(outputs, axis=1)
      for ind in self.final_delete_indices:
        del output_list[ind]
      outputs = tf.stack(output_list, axis=1)

      outputs = tf.squeeze(outputs, axis=-1)
      print(outputs.shape)
      result_stfts = tf.transpose(tf.signal.stft(outputs, self.window_length, self.hop_length, self.window_length), perm=[0,2,1])
      return tf.stack([tf.math.real(result_stfts),tf.math.imag(result_stfts)], axis=-1)
    
    return inputs

  def compute_output_shape(self, input_shape):
    temp = tf.random.normal((1, self.output_length, ))
    stft = tf.signal.stft(temp, self.window_length, self.hop_length, self.window_length)
    return tf.TensorShape([None, 512, stft.shape[1], 2])

class TimeStretch(Layer):
  def __init__(self, batch_size=BATCH_SIZE, signal_length=33216, segment_length=1000, overlap_length=50, hop_size=900, probability=15, hop_length=HOP_LENGTH, step_size=STEP_SIZE, window_length=WINDOW_LENGTH, message="TimeFold", **kwargs):
    super().__init__(self, probability, hop_length, step_size, window_length, message, **kwargs)
    self.batch_size = batch_size
    self.signal_length = signal_length
    self.segment_length = segment_length
    self.overlap_length = overlap_length
    self.hop_size = hop_size

  def prepare(self):
    if self.hop_size < 700:
      self.to_hop = randint(self.hop_size, self.hop_size+60)
    else:
      self.to_hop = randint(self.hop_size, 940) #preko 940 je pucalo

    print(self.message, "to hop:", self.to_hop)
    self.num_segments = (self.signal_length-self.segment_length)//self.to_hop+1
    self.output_length = self.num_segments*(self.segment_length-2*self.overlap_length)+(self.num_segments+1)*self.overlap_length

    self.coeffs_inc = []
    self.coeffs_dec = []
    self.coeffs_const = []

    coeff_inc = tf.pad(tf.range(1, self.overlap_length+1), tf.constant([self.segment_length-self.overlap_length, self.output_length-self.segment_length], shape=(1,2)))
    coeff_inc = tf.cast(coeff_inc, tf.float32)/self.overlap_length
    coeff_inc = tf.reshape(tf.tile(coeff_inc, tf.constant(self.batch_size, shape=(1,))), (self.batch_size,-1, 1))
    self.coeffs_inc.append(coeff_inc)

    coeff_dec = tf.pad(tf.range(self.overlap_length, limit=0, delta=-1), tf.constant([self.segment_length-self.overlap_length, self.output_length-self.segment_length], shape=(1,2)))
    coeff_dec = tf.cast(coeff_dec, tf.float32)/self.overlap_length
    coeff_dec = tf.reshape(tf.tile(coeff_dec, tf.constant(self.batch_size, shape=(1,))), (self.batch_size,-1, 1))
    self.coeffs_dec.append(coeff_dec)

    coeff_const = tf.pad(tf.ones(shape=(self.segment_length-self.overlap_length)), tf.constant([0, self.output_length-self.segment_length+self.overlap_length], shape=(1,2)))
    coeff_const = tf.cast(coeff_const, tf.float32)
    coeff_const = tf.reshape(tf.tile(coeff_const, tf.constant(self.batch_size, shape=(1,))), (self.batch_size,-1, 1))
    self.coeffs_const.append(coeff_const)

    for ind in range(self.num_segments-2):
      
      coeff_inc = tf.pad(tf.range(1, self.overlap_length+1), tf.constant([(ind+2)*(self.segment_length-self.overlap_length), self.output_length-(ind+2)*(self.segment_length-self.overlap_length)-self.overlap_length],shape=(1,2)))
      coeff_inc = tf.cast(coeff_inc, tf.float32)/self.overlap_length
      coeff_inc = tf.reshape(tf.tile(coeff_inc, tf.constant(self.batch_size, shape=(1,))), (self.batch_size,-1, 1))
      self.coeffs_inc.append(coeff_inc)

      coeff_dec = tf.pad(tf.range(self.overlap_length, limit=0, delta=-1), tf.constant([(ind+2)*(self.segment_length-self.overlap_length), self.output_length-(ind+2)*(self.segment_length-self.overlap_length)-self.overlap_length],shape=(1,2)))
      coeff_dec = tf.cast(coeff_dec, tf.float32)/self.overlap_length
      coeff_dec = tf.reshape(tf.tile(coeff_dec, tf.constant(self.batch_size, shape=(1,))), (self.batch_size,-1, 1))
      self.coeffs_dec.append(coeff_dec)

      coeff_const = tf.pad(tf.ones(shape=(self.segment_length-2*self.overlap_length)), tf.constant([self.overlap_length+(ind+1)*(self.segment_length-self.overlap_length), self.output_length-(ind+2)*(self.segment_length-self.overlap_length)], shape=(1,2)))
      coeff_const = tf.cast(coeff_const, tf.float32)
      coeff_const = tf.reshape(tf.tile(coeff_const, tf.constant(self.batch_size, shape=(1,))), (self.batch_size,-1, 1))
      self.coeffs_const.append(coeff_const)

    coeff_const = tf.pad(tf.ones(shape=(self.segment_length-self.overlap_length)), tf.constant([self.output_length-self.segment_length+self.overlap_length, 0], shape=(1,2)))
    coeff_const = tf.cast(coeff_const, tf.float32)
    coeff_const = tf.reshape(tf.tile(coeff_const, tf.constant(self.batch_size, shape=(1,))), (self.batch_size,-1, 1))
    self.coeffs_const.append(coeff_const)

  def call(self, inputs):
    random_number = randint(1, 100)
    if random_number <= self.probability:
      print(self.message)
      self.prepare()
      stfts = tf.complex(inputs[:,:,:,0], inputs[:,:,:,1])
      signals = tf.signal.inverse_stft(tf.transpose(stfts, perm=[0,2,1]), self.window_length, self.hop_length, self.window_length, window_fn=tf.signal.inverse_stft_window_fn(self.hop_length))
      signals = tf.expand_dims(signals, axis=-1)

      signal_length = signals.shape[1]

      sigs = []
      sig = tf.pad(signals, tf.constant([[0,0], [0, self.output_length-signal_length],[0,0]], shape=(3,2)))
      sigs.append(sig)

      for ind in range(self.num_segments-1):
        unstacked_sig = tf.unstack(sig, axis=1)
        for i in range(self.segment_length-self.to_hop-self.overlap_length):
          del unstacked_sig[self.output_length-i-1]

        sig = tf.pad(tf.stack(unstacked_sig, axis=1), tf.constant([[0,0], [self.segment_length-self.to_hop-self.overlap_length, 0], [0,0]], shape=(3,2)))
        sigs.append(sig)

      outs = []
      for ind in range(self.num_segments):
        if ind==0:
            out1 = Multiply()([self.coeffs_dec[ind], sigs[ind]])
            out2 = Multiply()([self.coeffs_const[ind], sigs[ind]])
            out = Add()([out1, out2])
            outs.append(out)
        elif ind==self.num_segments-1:     
            out1 = Multiply()([self.coeffs_inc[ind-1], sigs[ind]])
            out2 = Multiply()([self.coeffs_const[ind], sigs[ind]])
            out = Add()([out1, out2])
            outs.append(out)
        else:
            out1 = Multiply()([self.coeffs_dec[ind], sigs[ind]])
            out2 = Multiply()([self.coeffs_inc[ind-1], sigs[ind]])
            out3 = Multiply()([self.coeffs_const[ind], sigs[ind]])
            out = Add()([out1, out2, out3])
            outs.append(out)

      outputs = Add()(outs)
    
      outputs = tf.squeeze(outputs, axis=-1)
      print(outputs.shape)
      result_stfts = tf.transpose(tf.signal.stft(outputs, self.window_length, self.hop_length, self.window_length), perm=[0,2,1])
      return tf.stack([tf.math.real(result_stfts),tf.math.imag(result_stfts)], axis=-1)
    
    return inputs
    
  def compute_output_shape(self, input_shape):
    temp = tf.random.normal((1, self.output_length, ))
    stft = tf.signal.stft(temp, self.window_length, self.hop_length, self.window_length)
    return tf.TensorShape([None, 512, stft.shape[1], 2])   



class Attacks(Layer):
  def __init__(self, attack_probability=0.75, test=False, validation=False, attack_no=1, **kwargs):
    super(Attacks, self).__init__(**kwargs)
    self.attack_probability = attack_probability
    self.test = test
    self.validation = validation
    self.attack_no = attack_no
    
    self.frequency_filter = FrequencyFilter(cutoff_frequency=4000)
    self.additive_noise = AdditiveNoise(probability=100, message="AdditiveNoise1", dynamic=True)
    self.lowpass_filter = ButterworthFilter(probability=100, butterworth=BUTTERWORTH, message="ButterworthFilter1", dynamic=True)
    self.flip_samples = FlipSamples(probability=100, samples_to_flip=1600, message="FlipSamples1", dynamic=True)
    self.sample_cutting = SampleCutting(probability=100, samples_to_cut=1600, dynamic=True)
    self.time_fold = TimeFold(probability=100, hop_size=1050, overlap_length=25, message="TimeFold1", dynamic=True)
    self.time_stretch = TimeStretch(probability=100, hop_size=920, message="TimeStretch1", dynamic=True)
    self.delay = Delay(probability=100, delay=3200, message="Echo1", dynamic=True)
    self.downsampling = Downsampling(probability=100, factor=2, message="Downsampling", dynamic=True)
    self.upsampling = Upsampling(probability=100, factor=2, message="Upsampling", dynamic=True)

  def call(self, inputs, step):

    attack_index = 0
    random_number = randint(1, 100)
    if self.validation is True:
        random_number = 0

    if self.test is False:
      if random_number <= self.attack_probability*100:
        if step > 1400 and step <= 2800:
          attack_index = randint(1, 1)
        elif step > 2800 and step <= 7000:
          attack_index = randint(2, 2)
        elif step > 7000:
          attack_index = randint(1, 9)
    else:
      attack_index = self.attack_no
  
    if attack_index == 1:  
      out = self.lowpass_filter(inputs)
    elif attack_index == 3:
      out = self.downsampling(inputs)      
    elif attack_index == 2:
      out = self.sample_cutting(inputs)
    elif attack_index == 4: 
      out = self.flip_samples(inputs)
    elif attack_index == 5:
      out = self.time_fold(inputs)
    elif attack_index == 6:
      out = self.time_stretch(inputs)
    elif attack_index == 7:
      out = self.delay(inputs)
    elif attack_index == 8:
      out = self.upsampling(inputs)
    elif attack_index == 9:
      out = self.additive_noise(inputs)
    else:
      out = inputs

    if attack_index>1:
      random_number = 0
      if self.validation is True:
        random_number = randint(1,100)

      if random_number <= 75:
        out = self.lowpass_filter(out)

    return out
  
  def compute_output_shape(self, input_shape):
    return tf.TensorShape([None, 512, None, 2])