from distutils.command.config import config
import tensorflow as tf
import numpy as np
from attacks import Attacks
from config import BATCH_SIZE, NUM_BITS, WINDOW_LENGTH, HOP_LENGTH
from train_desync import generator, message_pool
from utils import snr
from pesq import pesq, PesqError


def test_step(embedder, detector, batch, test=False, validation=False, attack_no=None):
  input_stfts, input_messages = batch

  embedder_output, isWatermarked = embedder(batch, training=False)

  groundtruth_messages = input_messages
  if isWatermarked is False:
    print("Not watermarked")
    groundtruth_messages = tf.zeros(input_messages.shape)
  elif len(np.where((message_pool==input_messages[0,0,0]).all(axis=1))[0])==0:
    groundtruth_messages = tf.zeros(input_messages.shape)
    print("Random watermark")
  else:
    print("Watermark from pool")

  attacks = Attacks(test=test, validation=validation, attack_no=attack_no)
  attacks_output = attacks([embedder_output, isWatermarked])
  detector_output = detector(attacks_output)
  output_messages = tf.where(tf.greater_equal(detector_output, 0.5), 1, 0)
  groundtruth_messages = tf.cast(tf.squeeze(tf.slice(groundtruth_messages, [0, 0, 0, 0], [BATCH_SIZE, 1, 1, NUM_BITS])), tf.int32)
  mask = tf.where(tf.equal(output_messages, groundtruth_messages), 1, 0).numpy()
  
  
  total_pesq = 0
  total_snr = 0
  count = 0
  remove_indices = []

  input_stfts = tf.complex(input_stfts[:,:,:,0], input_stfts[:,:,:,1])      
  input_signals = tf.signal.inverse_stft(tf.transpose(input_stfts, perm=[0,2,1]), WINDOW_LENGTH, HOP_LENGTH, WINDOW_LENGTH, window_fn=tf.signal.inverse_stft_window_fn(HOP_LENGTH))

  output_stfts = tf.complex(embedder_output[:,:,:,0], embedder_output[:,:,:,1])      
  output_signals = tf.signal.inverse_stft(tf.transpose(output_stfts, perm=[0,2,1]), WINDOW_LENGTH, HOP_LENGTH, WINDOW_LENGTH, window_fn=tf.signal.inverse_stft_window_fn(HOP_LENGTH))

  if isWatermarked is True: 
    for i in np.arange(tf.shape(batch[0])[0]):
      curr_pesq = pesq(sr, input_signals[i].numpy(), output_signals[i].numpy(), 'nb', on_error=PesqError.RETURN_VALUES)
      if curr_pesq != -1: # isNan test
        total_pesq += curr_pesq
        total_snr += snr(input_signals[i], output_signals[i])
        count += 1
      else:
        remove_indices.append(i)

  mask = np.delete(mask, remove_indices, axis=0).astype(float)

  return np.sum(mask), total_pesq, total_snr, count


def test(embedder, detector, it, test=False, validation=False, attack_no=None, verbose=False):
  print("Testing...")
  total_acc = 0
  total_pesq = 0
  total_snr = 0
  count_pesq = 0
  count_acc = 0
  step = 1
  for batch in generator(it):
    batch_acc, batch_pesq, batch_snr, batch_count = test_step(embedder, detector, batch, test, validation, attack_no)
    if verbose is True:
      # print("Message index:", np.where((message_pool==batch[1][0,0,0]).all(axis=1))[0][0])
      if batch_count > 0:
        print("Average batch", step, "accuracy:", (batch_acc/(tf.cast(batch_count, dtype='float32')*NUM_BITS)).numpy(), "pesq:", (batch_pesq/tf.cast(batch_count, dtype='float32')).numpy(), "snr:", (batch_snr/tf.cast(batch_count, dtype='float32')).numpy())
      else:
        print("Average batch", step, "accuracy:", (batch_acc/(tf.cast(BATCH_SIZE, dtype='float32')*NUM_BITS)).numpy())

    total_acc += batch_acc
    if batch_count == 0:
      count_acc += BATCH_SIZE
    else:
      count_acc += batch_count
      total_pesq += batch_pesq
      total_snr += batch_snr
      count_pesq += batch_count

    step += 1
    
  return total_acc/(count_acc*NUM_BITS), total_pesq/count_pesq, total_snr/count_pesq