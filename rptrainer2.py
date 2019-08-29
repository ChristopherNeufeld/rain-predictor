#! /usr/bin/python3

# Here we go again.  Training the neural network.

import rpreddtypes
import argparse
import random

import tensorflow as tf
# from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

import keras
from keras.layers import Input, Dense, Concatenate, LSTM
from keras.models import Sequential, Model

import sys
import numpy as np



def getDataVectors(sequence_file, path_file):
    pathmap = {}
    seqmap = {}
    seqlist = []
    with open(path_file, 'r') as ifile:
        for record in ifile:
            fields = record.split()
            seqno = int(fields[0])
            pathmap[seqno] = fields[1]

    with open(sequence_file, 'r') as ifile:
        for record in ifile:
            fields = record.split()
            seqno = int(fields[0])
            seqmap[seqno] = list(map(int, fields[4:]))
            seqlist.append(seqno)

    random.shuffle(seqlist)

    # Need to load the size of the data samples by loading one data
    # file up front
    probeseqno = seqlist[0]
    probefilename = pathmap[seqno]
    reader = rpreddtypes.RpBinReader()
    reader.read(probefilename)
    rpbo = reader.getPreparedDataObject()
    datasize = rpbo.getDataLength()

    rvalX = np.empty([len(seqlist), 6, datasize])
    rvalY = np.empty([len(seqlist), 10])

    for index in range(len(seqlist)):
        base_seqno = seqlist[index]
        for timestep in range(6):
            ts_seqno = base_seqno + timestep
            ts_filename = pathmap[ts_seqno]
            reader = rpreddtypes.RpBinReader()
            reader.read(ts_filename)
            rpbo = reader.getPreparedDataObject()
            rvalX[index][timestep] = np.asarray(rpbo.getPreparedData()) / 255

        rvalY[index] = np.asarray(seqmap[base_seqno])

    return rvalX, rvalY, datasize



### Main code entry point here


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
                    'to use.  A fraction will be retained for '
                    'validation.')
parser.add_argument('--savefile', type=str, dest='savefile',
                    help='The filename at which to save the '
                    'trained network parameters.  A suffix will be '
                    'applied to the name to avoid data '
                    'incompatibility.')
parser.add_argument('--validation-frac', type=float, dest='vFrac',
                    default = 0.2,
                    help = 'That fraction of the training set to '
                    'be set aside for validation rather than '
                    'training.')
parser.add_argument('--epochs', type=int, dest='nEpochs',
                    default = 100,
                    help = 'Set the number of epochs to train.')

args = parser.parse_args()


xvals = None
yvals = None
datasize = None

xvals, yvals, datasize = getDataVectors(args.trainingset, args.pathfile)
    

if args.Continue:
    if not args.savefile:
        print('You asked to continue by loading a previous state, '
              'but did not supply the savefile with the previous state.')
        sys.exit(1)

    mymodel = keras.models.load_model(args.savefile)
    
else:

    inputs1 = Input(batch_shape = (None, 6, datasize))
    
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
#                                     save_best_only = True,
#                                     monitor='val_loss',
#                                     verbose=1,
#                                     mode='auto', period=1)

print ('Training\n')
mymodel.fit(x = xvals, y = yvals, epochs = args.nEpochs, verbose=1,
            validation_split = args.vFrac, shuffle = True)


if args.savefile:
    print('Saving model\n')
    mymodel.save(args.savefile)
