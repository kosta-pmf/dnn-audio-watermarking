import tensorflow as tf
from tensorflow_addons.layers.adaptive_pooling import AdaptiveAveragePooling2D, AdaptiveMaxPooling2D
from tensorflow.keras.layers import Layer, Reshape

class PyramidPooling2D(Layer):
  def __init__(self, bins=[(16, 16), (4, 4), (1, 1)], padding=False, method="MAX", **kwargs):
    super(PyramidPooling2D, self).__init__(**kwargs)
    self.bins = bins
    if method == "MAX":
      self.pool_layers = [AdaptiveMaxPooling2D(bin) for bin in self.bins]
    else:
      self.pool_layers = [AdaptiveAveragePooling2D(bin) for bin in self.bins]

    self.padding = padding
  
  def call(self, inputs):
    input_shape = inputs.shape
    outputs = []
    for ind, bin in enumerate(self.bins):

      height_overflow = input_shape[1] % bin[0]
      width_overflow = 0
      if self.padding is False:
        new_input_height = input_shape[1] - height_overflow
        new_input_width = input_shape[2] - width_overflow

        new_inputs = inputs[:, :new_input_height, :new_input_width, :]
      else:
        pad_height = bin[0] - height_overflow
        pad_width = bin[1] - width_overflow
        new_inputs = tf.pad(inputs, tf.constant([[0, 0], [0, pad_height], [0, pad_width], [0, 0]]))

      out = self.pool_layers[ind](new_inputs)
      out = Reshape((bin[0]*bin[1], input_shape[-1]))(out)
      outputs.append(out)

    return tf.concat(outputs, axis=1)