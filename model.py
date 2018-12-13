# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""SlowFast_Network model for Keras.

# Reference:

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)

Adapted from code from keras.application.resnet50.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Conv3D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import GlobalAveragePooling3D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling3D
from tensorflow.python.keras.models import Model



def identity_block(input_tensor, kernel_size, filters, stage, block, path, non_degenerate_temporal_conv=False):
  """The identity block is the block that has no conv layer at shortcut.

  Arguments:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names

  Returns:
      Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if K.image_data_format() == 'channels_last':
    bn_axis = 4
  else:
    bn_axis = 1
  conv_name_base = str(path) + 'res' + str(stage) + block + '_branch'
  bn_name_base = str(path) + 'bn' + str(stage) + block + '_branch'

  if non_degenerate_temporal_conv == True:
      x = Conv3D(filters1, (3, 1, 1), padding='same', name=conv_name_base + '2a')(input_tensor)
      x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
      x = Activation('relu')(x)
  else:
      x = Conv3D(filters1, (1, 1, 1), name=conv_name_base + '2a')(input_tensor)
      x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
      x = Activation('relu')(x)

  x = Conv3D(
      filters2, kernel_size, padding='same', name=conv_name_base + '2b')(
          x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)

  x = Conv3D(filters3, (1, 1, 1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

  x = layers.add([x, input_tensor])
  x = Activation('relu')(x)
  return x


def conv_block(input_tensor, kernel_size, filters, stage, block, path,strides=(1, 2 ,2),non_degenerate_temporal_conv=False):
  """A block that has a conv layer at shortcut.

  Arguments:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Strides for the first conv layer in the block.

  Returns:
      Output tensor for the block.

  Note that from stage 3,
  the first conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well
  """
  filters1, filters2, filters3 = filters
  if K.image_data_format() == 'channels_last':
    bn_axis = 4
  else:
    bn_axis = 1
  conv_name_base = str(path) + 'res' + str(stage) + block + '_branch'
  bn_name_base = str(path) + 'bn' + str(stage) + block + '_branch'

  if non_degenerate_temporal_conv == True:
      x = Conv3D(
          filters1, (3, 1, 1), strides=strides, padding='same', name=conv_name_base + '2a')(
              input_tensor)
      x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
      x = Activation('relu')(x)
  else:
      x = Conv3D(
          filters1, (1, 1, 1), strides=strides, name=conv_name_base + '2a')(
              input_tensor)
      x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
      x = Activation('relu')(x)
  x = Conv3D(
      filters2, kernel_size, padding='same', name=conv_name_base + '2b')(
          x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)

  x = Conv3D(filters3, (1, 1 ,1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

  shortcut = Conv3D(
      filters3, (1, 1, 1), strides=strides, name=conv_name_base + '1')(
          input_tensor)
  shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

  x = layers.add([x, shortcut])
  x = Activation('relu')(x)
  return x


def lateral_connection(fast_res_block,slow_res_block,method = 'T_conv',alpha=8,beta=1/8):
    if method not in ['T_conv','T_sample','TtoC_sum','TtoC_concat']:
        raise ValueError("method should be one of ['T_conv','T_sample','TtoC_sum','TtoC_concat']")
    if method == 'T_conv':
        lateral = Conv3D(int(2*beta*int(fast_res_block.shape[4])),padding='same',kernel_size=(5, 1, 1),strides=(int(alpha), 1, 1))(fast_res_block)
        connection = Concatenate(axis=-1)([slow_res_block,lateral])

    return connection



def SlowFast_Network(clip_shape=[64,224,224,3],num_class = 400,alpha=8,beta=1/8,tau=16,method= 'T_conv'):
  """Instantiates the SlowFast_Network architecture.

  Arguments:
      clip_shape: video_clip_shape

      num_class: numbers of videos class

      alpha:  mentioned in paper
      beta:   mentioned in paper
      tau:    mentioned in paper
      method: one of ['T_conv','T_sample','TtoC_sum','TtoC_concat'] mentioned in paper

  Returns:
      A Keras model instance.


  Raises:
      ValueError: in case of invalid argument for `method`
  """

  clip_shape = clip_shape
  slow_input_shape = [int(clip_shape[0]/tau),clip_shape[1],clip_shape[2],clip_shape[3]]
  fast_input_shape = [int(slow_input_shape[0]*alpha),slow_input_shape[1],slow_input_shape[2],slow_input_shape[3]]
  print('slow_path_input_shape',slow_input_shape)
  print('fast_path_input_shape',fast_input_shape)
  slow_input = Input(shape=slow_input_shape)
  fast_input = Input(shape=fast_input_shape)
  if K.image_data_format() == 'channels_last':
    bn_axis = 4
  else:
    bn_axis = 1

  # ---fast pathway---
  x_fast = Conv3D(
      64, (5, 7, 7), strides=(1, 2, 2), padding='same', name='fast_conv1')(fast_input)
  x_fast = BatchNormalization(axis=bn_axis, name='fast_bn_conv1')(x_fast)
  x_fast = Activation('relu')(x_fast)
  pool1_fast = MaxPooling3D((1, 3, 3), strides=(1, 2, 2),name='poo1_fast')(x_fast)

  x_fast = conv_block(pool1_fast, [1, 3, 3], [int(64*beta), int(64*beta), int(256*beta)], stage=2, block='a', path='fast',strides=(1, 1, 1), non_degenerate_temporal_conv=True)
  x_fast = identity_block(x_fast, [1, 3, 3], [int(64*beta), int(64*beta), int(256*beta)], stage=2, path='fast', block='b', non_degenerate_temporal_conv=True)
  res2_fast = identity_block(x_fast, [1, 3, 3], [int(64*beta), int(64*beta), int(256*beta)], stage=2, path='fast', block='c', non_degenerate_temporal_conv=True)

  x_fast = conv_block(res2_fast, [1, 3, 3], [int(128*beta), int(128*beta), int(512*beta)], stage=3, path='fast', block='a', non_degenerate_temporal_conv=True)
  x_fast = identity_block(x_fast, [1, 3, 3], [int(128*beta), int(128*beta), int(512*beta)], stage=3, path='fast', block='b', non_degenerate_temporal_conv=True)
  x_fast = identity_block(x_fast, [1, 3, 3], [int(128*beta), int(128*beta), int(512*beta)], stage=3, path='fast', block='c', non_degenerate_temporal_conv=True)
  res3_fast = identity_block(x_fast, [1, 3, 3], [int(128*beta), int(128*beta), int(512*beta)], stage=3, path='fast', block='d', non_degenerate_temporal_conv=True)

  x_fast = conv_block(res3_fast, [1, 3, 3], [int(256*beta), int(256*beta), int(1024*beta)], stage=4, path='fast', block='a', non_degenerate_temporal_conv=True)
  x_fast = identity_block(x_fast, [1, 3, 3], [int(256*beta), int(256*beta), int(1024*beta)], stage=4, path='fast', block='b', non_degenerate_temporal_conv=True)
  x_fast = identity_block(x_fast, [1, 3, 3], [int(256*beta), int(256*beta), int(1024*beta)], stage=4, path='fast', block='c', non_degenerate_temporal_conv=True)
  x_fast = identity_block(x_fast, [1, 3, 3], [int(256*beta), int(256*beta), int(1024*beta)], stage=4, path='fast', block='d', non_degenerate_temporal_conv=True)
  x_fast = identity_block(x_fast, [1, 3, 3], [int(256*beta), int(256*beta), int(1024*beta)], stage=4, path='fast', block='e', non_degenerate_temporal_conv=True)
  res4_fast = identity_block(x_fast, [1, 3, 3], [int(256*beta), int(256*beta), int(1024*beta)], stage=4, path='fast', block='f', non_degenerate_temporal_conv=True)

  x_fast = conv_block(res4_fast, [1, 3, 3], [int(512*beta), int(512*beta), int(2048*beta)], stage=5, path='fast', block='a', non_degenerate_temporal_conv=True)
  x_fast = identity_block(x_fast, [1, 3, 3], [int(512*beta), int(512*beta), int(2048*beta)], stage=5, path='fast', block='b', non_degenerate_temporal_conv=True)
  res5_fast = identity_block(x_fast, [1, 3, 3], [int(512*beta), int(512*beta), int(2048*beta)], stage=5, path='fast', block='c', non_degenerate_temporal_conv=True)

  # ---slow pathway---
  x = Conv3D(
      64, (1, 7, 7), strides=(1, 2, 2), padding='same', name='slow_conv1')(slow_input)
  x = BatchNormalization(axis=bn_axis, name='slow_bn_conv1')(x)
  x = Activation('relu')(x)
  pool1 = MaxPooling3D((1, 3, 3), strides=(1, 2, 2),name='poo1_slow')(x)
  pool1_conection = lateral_connection(pool1_fast,pool1,alpha=alpha,beta=beta)

  x = conv_block(pool1_conection, [1, 3, 3], [64, 64, 256], stage=2, block='a', strides=(1, 1 ,1), path='slow')
  x = identity_block(x, [1, 3, 3], [64, 64, 256], stage=2, block='b', path='slow')
  res2 = identity_block(x, [1, 3, 3], [64, 64, 256], stage=2, block='c', path='slow')
  res2_conection = lateral_connection(res2_fast,res2,alpha=alpha,beta=beta)

  x = conv_block(res2_conection, [1, 3, 3], [128, 128, 512], stage=3, block='a', path='slow')
  x = identity_block(x, [1, 3, 3], [128, 128, 512], stage=3, block='b', path='slow')
  x = identity_block(x, [1, 3, 3], [128, 128, 512], stage=3, block='c', path='slow')
  res3 = identity_block(x, [1, 3, 3], [128, 128, 512], stage=3, block='d', path='slow')
  res3_conection = lateral_connection(res3_fast,res3,alpha=alpha,beta=beta)

  x = conv_block(res3_conection, [1, 3, 3], [256, 256, 1024], stage=4, block='a', path='slow', non_degenerate_temporal_conv=True)
  x = identity_block(x, [1, 3, 3], [256, 256, 1024], stage=4, block='b', path='slow', non_degenerate_temporal_conv=True)
  x = identity_block(x, [1, 3, 3], [256, 256, 1024], stage=4, block='c', path='slow', non_degenerate_temporal_conv=True)
  x = identity_block(x, [1, 3, 3], [256, 256, 1024], stage=4, block='d', path='slow', non_degenerate_temporal_conv=True)
  x = identity_block(x, [1, 3, 3], [256, 256, 1024], stage=4, block='e', path='slow', non_degenerate_temporal_conv=True)
  res4 = identity_block(x, [1, 3, 3], [256, 256, 1024], stage=4, block='f', path='slow', non_degenerate_temporal_conv=True)
  res4_conection = lateral_connection(res4_fast,res4,alpha=alpha,beta=beta)

  x = conv_block(res4_conection, [1, 3, 3], [512, 512, 2048], stage=5, block='a', path='slow', non_degenerate_temporal_conv=True)
  x = identity_block(x, [1, 3, 3], [512, 512, 2048], stage=5, block='b', path='slow', non_degenerate_temporal_conv=True)
  res5 = identity_block(x, [1, 3, 3], [512, 512, 2048], stage=5, block='c', path='slow', non_degenerate_temporal_conv=True)

  fast_output = GlobalAveragePooling3D(name='avg_pool_fast')(res5_fast)
  slow_output = GlobalAveragePooling3D(name='avg_pool_slow')(res5)
  concat_output = Concatenate(axis=-1)([slow_output,fast_output])
  output = Dense(num_class,activation='softmax',name = 'fc')(concat_output)

  # Create model.
  inputs = [slow_input,fast_input]
  output = output
  model = Model(inputs, output, name='slowfast_resnet50')

  return model
