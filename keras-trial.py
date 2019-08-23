#! /usr/bin/python3

# Figure out how to make our through-time siamese network with shared
# weights

import keras
from keras.layers import Input, Dense, Concatenate, LSTM
from keras.models import Sequential, Model

import sys
import numpy as np


ring0_pixels = 5
ring1_pixels = 4
timesteps = 2
batch_size = 128
ring0_module_nodes_0 = 4
ring0_module_nodes_1 = 3
ring1_module_nodes_0 = 4
ring1_module_nodes_1 = 3
synth_layer_nodes = 3
num_outputs = 3




ring00 = Input(batch_shape=(batch_size, timesteps, ring0_pixels))
ring01 = Input(batch_shape=(batch_size, timesteps, ring0_pixels))
ring10 = Input(batch_shape=(batch_size, timesteps, ring1_pixels))
ring11 = Input(batch_shape=(batch_size, timesteps, ring1_pixels))

ring0_model = Sequential()
ring0_model.add(Dense(ring0_module_nodes_0))
ring0_model.add(Dense(ring0_module_nodes_1))

ring1_model = Sequential()
ring1_model.add(Dense(ring1_module_nodes_0))
ring1_model.add(Dense(ring1_module_nodes_1))

scanned00 = ring0_model(ring00)
scanned01 = ring0_model(ring01)
scanned10 = ring1_model(ring10)
scanned11 = ring1_model(ring11)


aggregated = Concatenate()([scanned00, scanned01, scanned10, scanned11])


time_layer = LSTM(3, stateful=False, return_sequences=True)(aggregated)

synth_layer = Dense(synth_layer_nodes)(time_layer)
output_layer = Dense(num_outputs)(synth_layer)

model = Model(inputs=[ring00, ring01,
                      ring10, ring11],
              outputs=[output_layer])
# model.compile(optimizer='SGD', loss=keras.losses.mean_squared_error)
