#! /usr/bin/python3

# Try to write a neural network to distinguish real from phantom rain.

# Phantom rain looks different from real rain, particularly in the
# region surrounding the radar station.  Ottawa lies within the radius
# where phantom rain can appear, so we need to know when active rain
# pixels are actually indicative of rain.

import argparse
import numpy
import rpreddtypes
import os
import keras
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Model

parser = argparse.ArgumentParser(description='Train the phantom '
                                 'classification network.')
parser.add_argument('--training-set', type=str, dest='training',
                    required=True,
                    help='A file of pathnames, and then letter '
                    '\'y\' for phantom rain, \'n\' for non-phantom '
                    '(i.e. real) rain.')
parser.add_argument('--validation-set', type=str, dest='validation',
                    required=True,
                    help='A file of pathnames, and then letter '
                    '\'y\' for phantom rain, \'n\' for non-phantom '
                    '(i.e. real) rain.')
parser.add_argument('--examination-box', type=list, dest='bounds',
                    default=[240, 320, 160, 240],
                    help='Bounds of the region to pass to the network.'
                    '  They are [minCol, maxCol, minRow, maxRow].')
parser.add_argument('--epochs', type=int, dest='nEpochs',
                    default = 100,
                    help = 'Set the number of epochs to train.')
parser.add_argument('--dense-layer-nodes', type=int, dest='densenodes',
                    default = 100,
                    help = 'Set the number of nodes in the synthesis layer.')
parser.add_argument('--name', type=str, dest='name',
                    required=True,
                    help='A name to distinguish this run.  It '
                    'will be used to construct filenames for '
                    'detailed logging.')

args = parser.parse_args()



def loadData(pathname, bounds):

    if os.path.exists(pathname + '-saved.npz'):
        container = numpy.load(pathname + '-saved.npz')
        return container['rvalX'], container['rvalY']
    
    minRow = bounds[2]
    minCol = bounds[0]
    numRows = bounds[3] - bounds[2] + 1
    numCols = bounds[1] - bounds[0] + 1
    records = []
    with open(pathname, 'r') as ifile:
        records = ifile.readlines()

    rvalX = numpy.zeros((len(records), 1, numRows, numCols))
    rvalY = numpy.zeros((len(records)))
    index = 0
    for r in records:
        isPhantom = r[-2:-1]
        if isPhantom == 'y':
            rvalY[index] = 1
        elif isPhantom == 'n':
            rvalY[index] = 0
        else:
            os.exit(1)

        binfilename = r[:-4] + '.bin'
        
        reader = rpreddtypes.RpBinReader()
        reader.read(binfilename)
        mrv = reader.getMaxRainval()
        rawdat = reader.getScaledObject(1).getNumpyArrayMax()
        for row in range(numRows):
            for col in range(numCols):
                # normalizing on the range [-1,1]
                rvalX[index, 0, row, col] = (rawdat[minRow + row, minCol + col] / mrv - 0.5) * 2

        index += 1

    numpy.savez(pathname + '-saved.npz', rvalX = rvalX, rvalY = rvalY)
    return rvalX, rvalY
        

trainingX, trainingY = loadData(args.training, args.bounds)
validateX, validateY = loadData(args.validation, args.bounds)

numrows = args.bounds[3] - args.bounds[2] + 1
numcols = args.bounds[1] - args.bounds[0] + 1


inputs = Input(batch_shape = (None, 1, numrows, numcols))
convlayer1 = Conv2D(filters=64, kernel_size=3, data_format='channels_first', activation='relu')(inputs)
pool1 = MaxPooling2D()(convlayer1)
convlayer2 = Conv2D(filters=32, kernel_size=3, activation='relu')(pool1)
flat = Flatten()(convlayer2)
synthlayer = Dense(args.densenodes, activation='relu')(flat)
outlayer = Dense(1, activation='sigmoid')(synthlayer)

mymodel = Model(inputs = inputs, outputs = outlayer)
mymodel.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])




basename = 'phantomnet_{}.'.format(args.densenodes)
cb1 = keras.callbacks.ModelCheckpoint(basename,
                                      save_weights_only=False,
                                      save_best_only = True,
                                      verbose=1,
                                      mode='auto', period=1)


history = mymodel.fit(x = trainingX, y = trainingY, epochs = args.nEpochs,
                      validation_data = [validateX, validateY],
                      verbose=1, batch_size=512, shuffle = True,
                      callbacks=[cb1])

for key in history.history.keys():
    filename='history_{}_{}'.format(args.densenodes, key)
    with open(filename, 'w') as ofile:
        ofile.write('\n'.join(str(val) for val in history.history[key]))
