#! /usr/bin/python3

# Try to write a neural network to distinguish real from phantom rain.

# Phantom rain looks different from real rain, particularly in the
# region surrounding the radar station.  Ottawa lies within the radius
# where phantom rain can appear, so we need to know when active rain
# pixels are actually indicative of rain.

import time
import argparse
import numpy
import rpreddtypes
import os
import keras
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, SpatialDropout2D, Lambda

from keras.models import Model

import matplotlib
from matplotlib import pyplot as plt


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
                    default=[240, 295, 185, 240],
                    help='Bounds of the region to pass to the network.'
                    '  They are [minCol, maxCol, minRow, maxRow].')
parser.add_argument('--epochs', type=int, dest='nEpochs',
                    default = 100,
                    help = 'Set the number of epochs to train.')
parser.add_argument('--dense-layer-nodes', type=int, dest='densenodes',
                    default = 100,
                    help = 'Set the number of nodes in the synthesis layer.')
parser.add_argument('--continue', dest='Continue',
                    action='store_true',
                    help = 'Whether to load the previously saved network '
                    'and continue for a certain number of epochs.')
parser.add_argument('--name', type=str, dest='name',
                    required=True,
                    help='A name to distinguish this run.  It '
                    'will be used to construct filenames for '
                    'detailed logging.')

args = parser.parse_args()

basename = 'phantomnet_{}.'.format(args.densenodes)


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

if not args.Continue:

    inputs = Input(batch_shape = (None, 1, numrows, numcols))
    convlayer1 = Conv2D(filters=32, kernel_size=3, data_format='channels_first',
                        activation='relu')(inputs)
    convlayer2 = Conv2D(filters=32, kernel_size=3, data_format='channels_first',
                        activation='relu')(convlayer1)
    invert1 = Lambda(lambda x : x * -1)(convlayer2)
    pool1 = MaxPooling2D(data_format='channels_first')(invert1)
    invert2 = Lambda(lambda x : x * -1)(pool1)
    drop1 = SpatialDropout2D(0.5)(invert2)

    convlayer3 = Conv2D(filters=64, kernel_size=3, data_format='channels_first',
                        activation='relu')(drop1)
    convlayer4 = Conv2D(filters=64, kernel_size=3, data_format='channels_first',
                        activation='relu')(convlayer3)
    pool2 = MaxPooling2D(data_format='channels_first')(convlayer4)
    drop2 = SpatialDropout2D(0.5)(pool2)

    flat = Flatten()(drop2)
    synthlayer = Dense(args.densenodes, 
                       activation='relu')(flat)
    outlayer = Dense(1, activation='sigmoid')(synthlayer)

    mymodel = Model(inputs = inputs, outputs = outlayer)
    mymodel.compile(loss='binary_crossentropy',
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])

else:
    mymodel = keras.models.load_model(basename)



if args.nEpochs > 0:
    cb1 = keras.callbacks.ModelCheckpoint(basename,
                                          monitor = 'val_acc',
                                          save_weights_only=False,
                                          save_best_only = True,
                                          verbose=1,
                                          mode='auto', period=1)


    history = mymodel.fit(x = trainingX, y = trainingY, epochs = args.nEpochs,
                          validation_data = [validateX, validateY],
                          verbose=1, batch_size=512, shuffle = True,
                          callbacks=[cb1])

    for key in history.history.keys():
        filename='histories/history_{}_{}'.format(args.name, key)
        with open(filename, 'w') as ofile:
            ofile.write('\n'.join(str(val) for val in history.history[key]))

truePositive = None
trueNegative = None
falsePositive = None
falseNegative = None


mymodel = keras.models.load_model(basename)
        
ypred = mymodel.predict(x = validateX)
for datapt in range(len(ypred)):
    if validateY[datapt] == 0:
        if ypred[datapt] >= 0.5:
            print('False positive: {}\n'.format(datapt))
            if not falsePositive:
                falsePositive = [ datapt, 'falsePositive' ]
        else:
            if not trueNegative:
                trueNegative = [ datapt, 'trueNegative' ]

    if validateY[datapt] == 1:
        if ypred[datapt] < 0.5:
            print('False negative: {}\n'.format(datapt))
            if not falseNegative:
                falseNegative = [ datapt, 'falseNegative' ]
        else:
            if not truePositive:
                truePositive = [ datapt, 'truePositive' ]


print ('trueNegative= {}\ntruePositive= {}\nfalseNegative= {}\nfalsePositive= {}\n'.format(trueNegative, truePositive, falseNegative, falsePositive))

# code adapted from
# https://github.com/gabrielpierobon/cnnshapes/blob/master/README.md

num2dlayers = 11
layer_outputs = [layer.output for layer in mymodel.layers[1:num2dlayers]]
activation_model = Model(inputs=mymodel.input, outputs=layer_outputs)

layer_names = []
for layer in mymodel.layers[1:num2dlayers]:
    layer_names.append(layer.name)

images_per_row = 16

for datapt in [ trueNegative, truePositive, falseNegative, falsePositive]:
    input = numpy.array(validateX[datapt[0]])[numpy.newaxis, :, :, :]
    name = datapt[1]

    activations = activation_model.predict(input)

    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps

        n_features = layer_activation.shape[1] # Number of features in the feature map
        size = layer_activation.shape[2] #The feature map has shape (1, n_features, size, size).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix

        display_grid = numpy.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 col * images_per_row + row,
                                                 :, :]
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = numpy.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, # Displays the grid
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()
