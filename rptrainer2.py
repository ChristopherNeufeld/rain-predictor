#! /usr/bin/python3

# Here we go again.  Training the neural network.

import rpreddtypes
import argparse
import random
import hashlib
import os

import tensorflow as tf
# from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

import keras
from keras.layers import Input, Dense, Concatenate, LSTM, BatchNormalization, Dropout
from keras.models import Sequential, Model

import sys
import numpy as np


# >>> input = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# >>> kernel = [0, 3]
# >>> pad = [[i, i] for i in kernel]
# >>> padded_input = numpy.pad(input, pad, "wrap")
# >>> padded_input
# array([[1, 2, 3, 1, 2, 3, 1, 2, 3],
#        [4, 5, 6, 4, 5, 6, 4, 5, 6],
#        [7, 8, 9, 7, 8, 9, 7, 8, 9]])




### Main code entry point here


geometry_layer_nodes = 80
lstm_module_nodes = 30
synth_layer_nodes = 30
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
parser.add_argument('--validation-set', type=str, dest='validationset',
                    required=True,
                    help='The file containing the validation set '
                    'to use.')
parser.add_argument('--savefile', type=str, dest='savefile',
                    required=True,
                    help='The filename at which to save the '
                    'trained network parameters.  A suffix will be '
                    'applied to the name to avoid data '
                    'incompatibility.')
parser.add_argument('--saved-vecs-dir', type=str, dest='savedvecs',
                    help='A directory to hold pre-loaded training '
                    'and validation data.  If files are present, '
                    'will load from there, otherwise it will load '
                    'as normal and save there.')
parser.add_argument('--holdout0', type=str, dest='holdout0',
                    help='The holdout dataset used for final '
                    'validation, no rain at present')
parser.add_argument('--holdout1', type=str, dest='holdout1',
                    help='The holdout dataset used for final '
                    'validation, raining at present')
parser.add_argument('--epochs', type=int, dest='nEpochs',
                    default = 100,
                    help = 'Set the number of epochs to train.')
parser.add_argument('--tensorboard', type=bool, dest='tensorboard',
                    default = False,
                    help = 'Whether to produce tensorboard outputs.')
parser.add_argument('--ignore-hash', type=bool, dest='nohash',
                    default = False,
                    help = 'Ignore unexpected hash values in '
                    'the input data.  Hash verification is used '
                    'to ensure that comparison runs all are '
                    'exposed to the same dataset.')
parser.add_argument('--set-optimizer', type=int, dest='optimizer',
                    default=-1,
                    help='Override the optimizer in the code.\n  '
                    '0 - SVD\n  1 - RMSprop\n  2 - Adagrad\n  '
                    '3 - Adadelta\n  4 - Adam\n  5 - Adamax\n  '
                    '6 - Nadam')
parser.add_argument('--name', type=str, dest='name',
                    required=True,
                    help='A name to distinguish this run.  It '
                    'will be used to construct filenames for '
                    'detailed logging.')

args = parser.parse_args()


xvals = None
yvals = None
mvals = None

vxvals = None
vyvals = None
vmvals = None
vdatasize = None
vnpts = None

datasize = None
npts = None
hashval = None

if args.nEpochs > 0:

    needload = True
    if args.savedvecs:
        if os.path.exists(args.savedvecs + '/savedvecs.npz'):
            needload = False
            container = np.load(args.savedvecs + '/savedvecs.npz')
            xvals = container['xvals']
            yvals = container['yvals']
            mvals = container['mvals']
            vxvals = container['vxvals']
            vyvals = container['vyvals']
            vmvals = container['vmvals']
            datasize = xvals.shape[2]

    if needload:
        xvals, yvals, mvals, datasize, npts, hashval = rpreddtypes.getDataVectors(args.trainingset, args.pathfile)

        if not args.nohash:
            if hashval != '968918db5a466fa9':
                print('Unexpected hash value {0}.  Input data may have changed.'
                      .format(hashval))
                sys.exit(1)

        vxvals, vyvals, vmvals, vdatasize, vnpts, vhashval = rpreddtypes.getDataVectors(args.validationset, args.pathfile)

        if args.savedvecs:
            if not os.path.exists(args.savedvecs):
                os.mkdir(args.savedvecs)

            np.savez(args.savedvecs + '/savedvecs.npz',
                     xvals = xvals, yvals = yvals, mvals = mvals,
                     vxvals = vxvals, vyvals = vyvals, vmvals = vmvals)


useoptimizer = keras.optimizers.RMSprop()
if args.optimizer == 0:
    useoptimizer = keras.optimizers.SGD()
elif args.optimizer == 1:
    useoptimizer = keras.optimizers.RMSprop()
elif args.optimizer == 2:
    useoptimizer = keras.optimizers.Adagrad()
elif args.optimizer == 3:
    useoptimizer = keras.optimizers.Adadelta()
elif args.optimizer == 4:
    useoptimizer = keras.optimizers.Adam()
elif args.optimizer == 5:
    useoptimizer = keras.optimizers.Adamax()
elif args.optimizer == 6:
    useoptimizer = keras.optimizers.Nadam()

    

if args.Continue:
    if not args.savefile:
        print('You asked to continue by loading a previous state, '
              'but did not supply the savefile with the previous state.')
        sys.exit(1)

    mymodel = keras.models.load_model(args.savefile)
    
else:

    inputs1 = Input(batch_shape = (None, 6, datasize))
    inputsM = Input(batch_shape = (None, 1))


    geometry_layer = Dense(geometry_layer_nodes, activation='relu')(inputs1)
    drop_layer1 = Dropout(rate=0.5)(geometry_layer)

    time_layer = LSTM(lstm_module_nodes, stateful = False,
                      activation='relu')(drop_layer1)

    drop_layer2 = Dropout(rate=0.5)(time_layer)

    concat = Concatenate()([drop_layer2, inputsM])

    synth_layer = Dense(synth_layer_nodes, activation='relu')(concat)


    output_layer = Dense(num_outputs, activation='sigmoid')(synth_layer)

    mymodel = Model(inputs=[inputs1, inputsM], outputs=[output_layer])

    mymodel.compile(loss='binary_crossentropy',
                    optimizer=useoptimizer)


if args.nEpochs > 0:

    cb1 = keras.callbacks.ModelCheckpoint('cb' + args.savefile,
                                          save_weights_only=False,
                                          save_best_only = True,
                                          verbose=1,
                                          mode='auto', period=1)
    cb2 = keras.callbacks.TensorBoard(log_dir="logs/" + args.name,
                                      histogram_freq=1,
                                      write_images=True)

    calllist = [ cb1 ]
    if args.tensorboard:
        calllist.append(cb2)

    history = mymodel.fit(x = [xvals, mvals], y = yvals, epochs = args.nEpochs,
                          validation_data = [[vxvals, vmvals], vyvals],
                          verbose=1, batch_size = 512,
                          shuffle = True, callbacks = calllist)

    histdir = "histories/" + args.name
    if not os.path.exists(histdir):
        os.makedirs(histdir)

    for key in history.history.keys():
        filename = histdir + '/' + key
        with open(filename, 'w') as ofile:
            ofile.write('\n'.join(str(val) for val in history.history[key]))
    


# mymodel = keras.models.load_model('cb' + args.savefile)
# vypred = mymodel.predict(x = [ vxvals, vmvals ])
# for i in range(len(vxvals)):
#     print('{0} TRUE {1}'.format(i, vyvals[i]))
#     print('{0} PRED {1}'.format(i, vypred[i]))

    
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

    if args.savefile:
        mymodel = keras.models.load_model('cb' + args.savefile)
    
    confusion = np.zeros((10, 2, 2), dtype=np.int64)

    if args.holdout0:
        # This is the branch when it is not currently raining, but it
        # will rain soon.

        hxvals, hyvals, hjunk0, hjunk1, hnpts, hjunk2 = rpreddtypes.getDataVectors(args.holdout0, args.pathfile)

        willRainIn2 = 0
        predWillRainIn1or2 = 0
        willRainIn3plus = 0
        predWillRainIn3plus = 0
        
        hypred = mymodel.predict(x = hxvals)
        for datapt in range(hnpts):

            if hyvals[datapt, 0] == 0 and hyvals[datapt, 2] == 1:
                willRainIn2 += 1
                if hypred[datapt, 0] >= 0.5 or hypred[datapt, 2] >= 0.5:
                    predWillRainIn1or2 += 1

            if hyvals[datapt, 0] == 0 and hyvals[datapt, 2] == 0:
                willRainIn3plus += 1
                if hypred[datapt, 4] >= 0.5 or hypred[datapt, 6] >= 0.5 or hypred[datapt, 8] >= 0.5:
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

        print('RPRED:  Rain starting in 1-2 hours, warning rate= {0} / {1}: {2}'
              .format(predWillRainIn1or2, willRainIn2,
                      predWillRainIn1or2 / willRainIn2))
        print('RPRED:  Rain starting in 3-6 hours, warning rate= {}'
              .format(predWillRainIn3plus / willRainIn3plus))
        
        
    if args.holdout1:
        hxvals, hyvals, hjunk0, hjunk1, hnpts, hjunk2 = rpreddtypes.getDataVectors(args.holdout1, args.pathfile)

        willStopIn1 = 0
        predWillStopIn1 = 0
        willStopIn3plus = 0
        predWillStopIn3plus = 0

        hypred = mymodel.predict(x = hxvals)
        for datapt in range(hnpts):

            if hyvals[datapt, 0] == 0:
                willStopIn1 += 1
                if hypred[datapt, 0] < 0.5:
                    predWillStopIn1 += 1

            if hyvals[datapt, 0] == 1 and hyvals[datapt, 2] == 1:
                willStopIn3plus += 1
                if ( ( hypred[datapt, 0] >= 0.5
                       and hypred[datapt, 2] >= 0.5 )
                     and (hypred[datapt, 4] < 0.5
                          or hypred[datapt, 6] < 0.5
                          or hypred[datapt, 8] < 0.5)):
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
            

    print('Total confusion= {0}'.format(confusion))
