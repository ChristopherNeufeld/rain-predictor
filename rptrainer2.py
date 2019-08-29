#! /usr/bin/python3

# Here we go again.  Training the neural network.

import rpreddtypes
import rpgenerator2
import argparse
import random

import tensorflow as tf
# from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

import keras
from keras.layers import Input, Dense, Concatenate, LSTM
from keras.models import Sequential, Model

import sys
import numpy as np






### Main code entry point here


defBatchSize = 512
lstm_module_nodes = 500
synth_layer_nodes = 300
num_outputs = 10


parser = argparse.ArgumentParser(description='Train the rain '
                                 'prediction network.')
parser.add_argument('--continue', dest='Continue',
                    action='store_true',
                    help='Whether to load a previous state and '
                    'continue training')
parser.add_argument('--pathfile', type=str, dest='pathfile',
                    required=True,
                    help='The file that maps sequence numbers to '
                    'the pathnames of the binary files.')
parser.add_argument('--training-set', type=str, dest='trainingset',
                    required=True,
                    help='The file containing the training set '
                    'to use.')
parser.add_argument('--veto-set', type=str, dest='vetoset',
                    help='If supplied, its argument is the name '
                    'of a file containing training set entries that '
                    'are to be skipped.')
parser.add_argument('--test-set', type=str, dest='testset',
                    required=True,
                    help='The test set used to detect overfitting '
                    'during training.')
parser.add_argument('--savefile', type=str, dest='savefile',
                    help='The filename at which to save the '
                    'trained network parameters.  A suffix will be '
                    'applied to the name to avoid data '
                    'incompatibility.')
parser.add_argument('--override-centre', type=list, dest='centre',
                    default=[240,239], help='Set a new location for '
                    'the pixel coordinates of the radar station')
parser.add_argument('--override-sensitive-region', type=list,
                    dest='sensitive',
                    default=[[264,204], [264,205], [265,204], [265,205]],
                    help='Set a new list of sensitive pixels')
parser.add_argument('--heavy-rain-index', type=int, dest='heavy',
                    default=3, help='Lowest index in the colour table '
                    'that indicates heavy rain, where 1 is the '
                    'lightest rain.')
parser.add_argument('--batchsize', type=int, dest='batchsize',
                    default=defBatchSize,
                    help='Override the batch size used for training.')

args = parser.parse_args()


trainGenerator = rpgenerator2.RPDataGenerator2(args.trainingset,
                                               args.pathfile,
                                               args.vetoset,
                                               args.centre,
                                               args.sensitive,
                                               args.heavy,
                                               args.batchsize,)

validationGenerator = rpgenerator2.RPDataGenerator2(args.testset,
                                                    args.pathfile,
                                                    args.vetoset,
                                                    args.centre,
                                                    args.sensitive,
                                                    args.heavy,
                                                    args.batchsize)

hashval = rpreddtypes.genhash(args.centre, args.sensitive, args.heavy)

if args.savefile:
    args.savefile = args.savefile + str(hashval)


if args.Continue:
    if not args.savefile:
        print('You asked to continue by loading a previous state, '
              'but did not supply the savefile with the previous state.')
        sys.exit(1)

    mymodel = keras.models.load_model(args.savefile)
    
else:

    nInputs1 = trainGenerator.getInputSize()
    inputs1 = Input(batch_shape = (args.batchsize, 6, nInputs1))
    
    time_layer = LSTM(lstm_module_nodes, stateful = False,
                      activation='relu')(inputs1)

    synth_layer = Dense(synth_layer_nodes, activation='relu')(time_layer)
    output_layer = Dense(num_outputs, activation='sigmoid')(synth_layer)

    mymodel = Model(inputs=[inputs1], outputs=[output_layer])

print('Compiling\n')
mymodel.compile(loss='binary_crossentropy', optimizer='sgd')
#                metrics=[tf.keras.metrics.FalsePositives(),
#                         tf.keras.metrics.FalseNegatives()])


# if args.savefile:
#     keras.callbacks.ModelCheckpoint(args.savefile, save_weights_only=False,
#                                     save_best_only = False,
#                                     verbose=1,
#                                     mode='auto', period=1)

print ('Training\n')
mymodel.fit_generator(generator = trainGenerator,
                      validation_data = validationGenerator,
                      use_multiprocessing = False,
                      epochs = 4,
                      shuffle=False)

if args.savefile:
    print('Saving model\n')
    mymodel.save(args.savefile)
