#! /usr/bin/python3

# Here we go.  Training the neural network.

import rpreddtypes
import rpgenerator
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
ring_module_nodes_0 = 1000
ring_module_nodes_1 = 5
bullseye_module_nodes_0 = 1000
bullseye_module_nodes_1 = 5
tripwire_module_nodes_0 = 40
tripwire_module_nodes_1 = 5
synth_layer_nodes = 200
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
parser.add_argument('--coarse-factor', type=int, dest='coarseness',
                    default=1,
                    help='Coarse scale the data, reducing its size '
                    'by this factor in each of X and Y.')
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


trainGenerator = rpgenerator.RPDataGenerator(args.trainingset,
                                             args.pathfile,
                                             args.vetoset,
                                             args.centre,
                                             args.sensitive,
                                             args.heavy,
                                             args.batchsize,
                                             args.coarseness)

validationGenerator = rpgenerator.RPDataGenerator(args.testset,
                                                  args.pathfile,
                                                  args.vetoset,
                                                  args.centre,
                                                  args.sensitive,
                                                  args.heavy,
                                                  args.batchsize,
                                                  args.coarseness)

hashval = rpreddtypes.genhash(args.centre, args.sensitive, args.heavy)
modsizes = trainGenerator.getModuleSizes()

if args.savefile:
    args.savefile = args.savefile + str(hashval)


if args.Continue:
    if not args.savefile:
        print('You asked to continue by loading a previous state, '
              'but did not supply the savefile with the previous state.')
        sys.exit(1)

    mymodel = keras.models.load_model(args.savefile)
    
else:

    inputslist = list(map(lambda x:
                          Input(batch_shape=(args.batchsize, 6, x)),
                          modsizes))

    evenmodels = []
    oddmodels = []

    for i in range(4):
        onemod = Sequential()
        onemod.add(Dense(ring_module_nodes_0, activation='relu'))
        onemod.add(Dense(ring_module_nodes_1, activation='relu'))
        evenmodels.append(onemod)
        onemod = Sequential()
        onemod.add(Dense(ring_module_nodes_0, activation='relu'))
        onemod.add(Dense(ring_module_nodes_1, activation='relu'))
        oddmodels.append(onemod)

    bullseyemodel = Sequential()
    bullseyemodel.add(Dense(bullseye_module_nodes_0, activation='relu'))
    bullseyemodel.add(Dense(bullseye_module_nodes_1, activation='relu'))

    tripwiremodel = Sequential()
    tripwiremodel.add(Dense(tripwire_module_nodes_0, activation='relu'))
    tripwiremodel.add(Dense(tripwire_module_nodes_1, activation='relu'))

    
    
        
    scanned = []
    for i in range(32):
        mtype = i % 2
        ringnum = i // 8
        if mtype == 0:
            scanned.append(evenmodels[ringnum](inputslist[i]))
        else:
            scanned.append(oddmodels[ringnum](inputslist[i]))

    scanned.append(bullseyemodel(inputslist[32]))
    scanned.append(tripwiremodel(inputslist[33]))

    aggregated = Concatenate()(scanned)

    time_layer = LSTM(6, stateful = False, activation='relu')(aggregated)

    synth_layer = Dense(synth_layer_nodes, activation='relu')(time_layer)
    output_layer = Dense(num_outputs, activation='sigmoid')(synth_layer)

    mymodel = Model(inputs=inputslist, outputs=[output_layer])

mymodel.compile(loss='binary_crossentropy', optimizer='sgd')
#                metrics=[tf.keras.metrics.FalsePositives(),
#                         tf.keras.metrics.FalseNegatives()])


if args.savefile:
    keras.callbacks.ModelCheckpoint(args.savefile, save_weights_only=False,
                                    save_best_only = False,
                                    verbose=1,
                                    mode='auto', period=1)

mymodel.fit_generator(generator=trainGenerator,
                      validation_data = validationGenerator,
                      shuffle=False)
