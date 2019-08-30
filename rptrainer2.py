#! /usr/bin/python3

# Here we go again.  Training the neural network.

import rpreddtypes
import argparse
import random
import hashlib

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
            seqmap[seqno] = list(map(int, fields[5:]))
            seqlist.append(seqno)

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

    hasher = hashlib.sha256()
    hasher.update(rvalX.data.tobytes())
    hasher.update(rvalY.data.tobytes())
    hashval = (hasher.hexdigest())[-16:]

    # Shuffle the vectors
    for i in range(len(seqlist)):
        newoffset = random.randint(i, len(seqlist) - 1)
        if newoffset == i:
            continue
        
        tmp = rvalX[i]
        rvalX[i] = rvalX[newoffset]
        rvalX[newoffset] = tmp

        tmp = rvalY[i]
        rvalY[i] = rvalY[newoffset]
        rvalY[newoffset] = tmp

    return rvalX, rvalY, datasize, len(seqlist), hashval



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
parser.add_argument('--holdout0', type=str, dest='holdout0',
                    help='The holdout dataset used for final '
                    'validation, no rain at present')
parser.add_argument('--holdout1', type=str, dest='holdout1',
                    help='The holdout dataset used for final '
                    'validation, raining at present')
parser.add_argument('--epochs', type=int, dest='nEpochs',
                    default = 100,
                    help = 'Set the number of epochs to train.')
parser.add_argument('--ignore-hash', type=bool, dest='nohash',
                    default = False,
                    help = 'Ignore unexpected hash values in '
                    'the input data.  Hash verification is used '
                    'to ensure that comparison runs all are '
                    'exposed to the same dataset.')

args = parser.parse_args()


xvals = None
yvals = None
datasize = None
npts = None
hashval = None

if args.nEpochs > 0:
    xvals, yvals, datasize, npts, hashval = getDataVectors(args.trainingset, args.pathfile)

if not args.nohash:
    if hashval != '048073a92f6dd2bb':
        print('Unexpected hash value {0}.  Input data may have changed.'
              .format(hashval))
        sys.exit(1)
    

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

    mymodel.compile(loss='binary_crossentropy', optimizer='sgd')


if args.nEpochs > 0:
    mymodel.fit(x = xvals, y = yvals, epochs = args.nEpochs, verbose=1,
                validation_split = args.vFrac, shuffle = True)


# My confusion matrix is  TP:  0,0
#                         FP:  0,1
#                         FN:  1,0
#                         TN:  1,1

hxvals = None
hyvals = None
hjunk1 = None
hnpts = None
hjunk2 = None

if args.holdout0 or args.holdout1:
    confusion = np.zeros((10, 2, 2), dtype=np.int64)

    if args.holdout0:
        # This is the branch when it is not currently raining, but it
        # will rain soon.

        hxvals, hyvals, hjunk1, hnpts, hjunk2 = getDataVectors(args.holdout0, args.pathfile)

        willRainIn2 = 0
        predWillRainIn1or2 = 0
        willRainIn3plus = 0
        predWillRainIn3plus = 0
        
        hypred = mymodel.predict(x = hxvals)
        for datapt in range(hnpts):

            if hyvals[datapt, 0] == 0 and hyvals[datapt, 2] == 1:
                willRainIn2 += 1
                if hpred[datapt, 0] >= 0.5 or hpred[datapt, 2] >= 0.5:
                    predWillRainIn1or2 += 1

            if hyvals[datapt, 0] == 0 and hyvals[datapt, 2] == 0:
                willRainIn3plus += 1
                if hpred[datapt, 4] >= 0.5 or hpred[datapt, 6] >= 0.5 or hpred[datapt, 8] >= 0.5:
                    predWillRainIn3plus += 1
            
            for bitnum in range(10):
                if hyvals[datapt, bitnum] == 0:
                    if hypred[datapt, bitnum] < 0.5:
                        confusion[bitnum, 0, 0] += 1       # True negative
                    else:
                        confusion[bitnum, 0, 1] += 1       # False positive
                else:
                    if hypred[datapt, bitnum] >= 0.5:
                        confusion[bitnum, 1, 1] += 1    # True positive
                    else:
                        confusion[bitnum, 1, 0] += 1    # False negative

        print('RPRED:  Rain starting in 1-2 hours, warning rate= {}'
              .format(predWillRainIn1or2 / willRainIn2))
        print('RPRED:  Rain starting in 3-6 hours, warning rate= {}'
              .format(predWillRainIn3plus / willRainIn3plus))
        
        
    if args.holdout1:
        hxvals, hyvals, hjunk1, hnpts, hjunk2 = getDataVectors(args.holdout1, args.pathfile)

        willStopIn1 = 0
        predWillStopIn1 = 0
        willStopIn3plus = 0
        predWillStopIn3plus = 0

        hypred = mymodel.predict(x = hxvals)
        for datapt in range(hnpts):

            if hyvals[datapt, 0] == 0:
                willStopIn1 += 1
                if hypred[datapt, 0] >= 0.5:
                    predWillStopIn1 += 1

            if hyvals[datapt, 0] == 1 and hyvals[datapt, 2] == 1:
                willStopIn3plus += 1
                if ( ( hypred[datapt, 0] < 0.5
                       and hypred[datapt, 2] < 0.5 )
                     and (hypred[datapt, 4] >= 0.5
                          or hypred[datapt, 6] >= 0.5
                          or hypred[datapt, 8] >= 0.5)):
                    predWillStopIn3plus += 1
                    
            for bitnum in range(10):
                if hyvals[datapt, bitnum] == 0:
                    if hypred[datapt, bitnum] < 0.5:
                        confusion[bitnum, 0, 0] += 1       # True negative
                    else:
                        confusion[bitnum, 0, 1] += 1       # False positive
                else:
                    if hypred[datapt, bitnum] > 0.5:
                        confusion[bitnum, 1, 1] += 1    # True positive
                    else:
                        confusion[bitnum, 1, 0] += 1    # False negative

        print('RPRED:  Rain stopping in next hour, warning rate= {}'
              .format(predWillStopIn1 / willStopIn1))
        print('RPRED:  Rain stopping in 3-6 hours, warning rate= {}'
              .format(predWillStopIn3plus / willStopIn3plus))
            

    print('Total confusion= {1}'.format(confusion))



if args.savefile:
    print('Saving model\n')
    mymodel.save(args.savefile)
