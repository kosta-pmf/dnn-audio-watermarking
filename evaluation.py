from config import BATCH_SIZE, FS, HOP_LENGTH, NUM_BITS, WINDOW_LENGTH
import tensorflow as tf
import numpy as np
from pypesq import pesq
import librosa
from utils import snr
from utils import generator
from tensorflow import keras

def reconstruct_from_stft(example):
  stft = tf.complex(example[:,:,0], example[:,:,1])
  signal = librosa.core.istft(stft.numpy(), hop_length=HOP_LENGTH, win_length=WINDOW_LENGTH-1, center=True)
  return signal

def test_step(model, batch):
  input_stfts, input_messages = batch
  
  encoder_output, attacks_output, decoder_output = model(batch, training=False)
  output_messages = tf.where(tf.greater_equal(decoder_output, 0.5), 1, 0)
  input_messages = tf.cast(tf.squeeze(tf.slice(input_messages, [0, 0, 0, 0], [BATCH_SIZE, 1, 1, NUM_BITS])), tf.int32)
  mask = tf.where(tf.equal(output_messages, input_messages), 1, 0).numpy()
  
  total_pesq = 0
  total_snr = 0
  count = 0
  remove_indices = []
  for i in np.arange(tf.shape(batch[0])[0]):
    input_signal = reconstruct_from_stft(np.squeeze(input_stfts[i].numpy()))
    output_signal = reconstruct_from_stft(np.squeeze(encoder_output[i].numpy()))
    curr_pesq = pesq(input_signal, output_signal, FS)
    if curr_pesq == curr_pesq: # isNan test
      total_pesq += curr_pesq
      total_snr += snr(input_signal, output_signal)
      count += 1
    else:
      remove_indices.append(i)

  mask = np.delete(mask, remove_indices, axis=0).astype(float)

  return np.sum(mask), total_pesq, total_snr, count

def test(model, it, verbose=False):
    print("Testing...")
    total_acc = 0
    total_pesq = 0
    total_snr = 0
    count = 0
    step = 1
    for batch in generator(it):
        batch_acc, batch_pesq, batch_snr, batch_count = test_step(model, batch)
        if verbose is True:
            print("Average batch", step, "accuracy:", (batch_acc/(tf.cast(batch_count, dtype='float32')*NUM_BITS)).numpy(), "pesq:", (batch_pesq/tf.cast(batch_count, dtype='float32')).numpy(), "snr:", (batch_snr/tf.cast(batch_count, dtype='float32')).numpy())
        total_acc += batch_acc
        total_pesq += batch_pesq
        total_snr += batch_snr
        count += batch_count
        step += 1
    
    return total_acc/(count*NUM_BITS), total_pesq/count, total_snr/count

def restore_model(model_path):
    restored_model = keras.model.load_model(model_path)
    return restored_model