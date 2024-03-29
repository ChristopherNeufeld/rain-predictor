#! /usr/bin/python3

# This script will parse one or more of the intermediate binary files
# for the rain predictor and will produce a record on stdout.  A
# record consists of one line:

# <SEQ_NO> <FULL_PATH> <HASH> <N_ROTS> <RAIN_0> <HEAVY_RAIN_0> ...

# The SEQ_NO is a sequence number, the number of .gif files that ought
# to have been produced between 2015-01-01T00:00:00Z and now, assuming
# constant 10 minute intervals.  That's because we need an unbroken
# sequence of 6 hours of data for a training element, so checking for
# contiguous blocks of sequence numbers will be a quick determinant

# FULL_PATH is the pathname of the intermediate binary file.

# HASH is a 32-bit integer computed from the pixel coordinates of the
# radar station and of the rain-sensing pixels.  Represented as a
# 32-bit hex value with a leading 0x, it is there so we can
# distinguish training records for different areas of interest, if we
# should ever expand to that.

# N_ROTS is the number of rotations in this data.  '0' indicates only
# the unrotated set is present.  For numbers greater than 0, the
# rotations are assumed to be evenly divided over the circle, and
# count COUNTER-CLOCKWISE from the unrotated entry.

# RAIN_0 HEAVY_RAIN_0 is 0 or 1, 1 if there is any rain/heavy-rain in
# any of the rain-sensing pixels for this intermediate binary file, 0
# otherwise.  The _0 suffix indicates the unrotated set.  If rotated
# data is present, there will be pairs of records for each such
# rotation.

# IMPORTANT NOTE:
#
# the rotations of the rain-sensing region are counter-clockwise
# rotations.  That's because we're going to consider the rotations of
# the input set to be clockwise rotations, and rotating the rain
# clockwise means we have to rotate the sensing pixels
# counter-clockwise to get correct results.

# Also, remember we're using the same pixel numbering scheme as in the
# .gif file, the first index counts across a row, the second counts
# down rows.  This is a left-handed coordinate system, adjust the
# trigonometry appropriately.

import argparse
import sys
import os
import rpreddtypes
import numpy as np
import math
import keras


phantomRainNetwork = None


def rotate_pixel_CCW (pixel, centre, nDiv, rotNum):
    """
    Rotates the pixel counter-clockwise around the centre by rotNum *
    2 pi / nDiv.  Special case treatment for nDiv = 2 or 4, since
    D_2 and D_4 symmetry groups are contained within the square grid.
    rotNum counts up from 1, so must not exceed nRots
    """

    if nDiv == 1:
        return pixel.copy()

    if nDiv <= 0 or rotNum <= 0 or rotNum > nDiv:
        print('Invalid invocation of rotate_pixel_CCW.  '
              'nDiv={0} and rotNum={1}'.format(nDiv, rotNum))
        sys.exit(1)

    delta = pixel.copy()
    delta[0] -= centre[0]
    delta[1] -= centre[1]

    rval = pixel.copy()

    if nDiv == 2 or ( nDiv == 4 and rotNum == 2):
        rval[0] = centre[0] - delta[0]
        rval[1] = centre[1] - delta[1]
        return rval

    if nDiv == 4:
        if rotNum == 1:
            rval[0] = centre[0] + delta[1]
            rval[1] = centre[1] - delta[0]
            return rval
        if rotNum == 3:
            rval[0] = centre[0] - delta[1]
            rval[1] = centre[1] + delta[0]
            return rval

    # If we get here, it's a rotation not covered by the symmetry of
    # the grid.  Note that we lose the 1-1 mapping guarantee now.
    # It's possible for two different input pixels to map to the same
    # output pixel.

    theta = (np.pi * 2 / nDiv) * rotNum
    sinT = np.sin(theta)
    cosT = np.cos(theta)

    # left-handed rotation matrix
    rmat = np.array(((c, s), (-s, c)))
    deltavec = np.array(delta)
    deltavec = rmat.dot(deltavec)

    rval[0] = centre[0] + deltavec[0]
    rval[1] = centre[1] + deltavec[1]
    return rval


def rainPresent(binReader, sensitivePixels, bounds, heavyVal, seqno):
    """
    Returns a list of 2 integer elements.  The first indicates any
    rain at all in any of the sensitive pixels.  The second indicates
    rain above the threshold intensity for heavy rain.
    """

    rpbo = binReader.getScaledObject(1)
    mrv = binReader.getMaxRainval()
    maxSeen = 0
    anyRain = 0
    heavyRain = 0
    checkPhantom = True
    xoffset = rpbo.getXOffset()
    yoffset = rpbo.getYOffset()
    offset = [yoffset, xoffset]
    dataWidth = rpbo.getWidth()
    dataHeight = rpbo.getHeight()
    data = rpbo.getNumpyArrayMax()
    
    for pixel in sensitivePixels:
        pval = data[pixel[0] - yoffset][pixel[1] - xoffset]
        if pval > 0:
            anyRain = 1
            if pval > 1:
                checkPhantom = False
            if pval >= maxSeen:
                maxSeen = pval
        if maxSeen >= heavyVal:
            heavyRain = 1
            return [ 1, 1 ]

    if not anyRain:
        return [ 0, 0 ]

    if not checkPhantom:
        return [ anyRain, heavyRain ]

    # Have to check for phantom rain.

    minRow = bounds[2]
    minCol = bounds[0]
    numRows = bounds[3] - bounds[2] + 1
    numCols = bounds[1] - bounds[0] + 1

    clipped = np.zeros((1, 1, numRows, numCols))
    for row in range(numRows):
        for col in range(numCols):
            clipped[0, 0, row, col] = (data[minRow + row, minCol + col] / mrv - 0.5) * 2

    isPhantom = phantomRainNetwork.predict(x = clipped)
    if isPhantom[0] >= 0.5:
        return [ 0, 0 ]
    else:
        return [ anyRain, heavyRain ]
    



## Main execution begins here

parser = argparse.ArgumentParser(description='Build training sequence.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('ifilenames', type=str, metavar='filename',
                    nargs='+', help='Filenames to process')

parser.add_argument('--override-centre', type=list, dest='centre',
                    default=[239,240], help='Set a new location for '
                    'the pixel coordinates of the radar station')
parser.add_argument('--override-sensitive-region', type=list,
                    dest='sensitive',
                    default=[[204,264], [205,264], [204,265], [205,265]],
                    help='Set a new list of sensitive pixels.  '
                    'In row,col order')
parser.add_argument('--rotations', type=int, dest='rotations',
                    default=0, help='Number of synthetic data points '
                    'to create (via rotation) for each input data point')
parser.add_argument('--phantom-network', type=str, dest='phantomnet',
                    required=True,
                    help='Name of the saved neural network that will '
                    'determine whether rain is true or \'phantom\'.')
parser.add_argument('--bounds', type=list, dest='bounds',
                    default=[240, 295, 185, 240],
                    help='Bounds of the region to pass to the network.'
                    '  They are [minCol, maxCol, minRow, maxRow].')
parser.add_argument('--heavy-rain-index', type=int, dest='heavy',
                    default=3, help='Lowest index in the colour table '
                    'that indicates heavy rain, where 1 is the '
                    'lightest rain.')

args = parser.parse_args()


phantomRainNetwork = keras.models.load_model(args.phantomnet)


hashString = rpreddtypes.genhash(args.centre, args.sensitive, args.heavy)

for inputfile in args.ifilenames:
    rpReader = rpreddtypes.RpBinReader()
    rpReader.read(inputfile)
    seqno, junk1, junk2, junk3, junk4, junk5 = rpreddtypes.computeSequenceNumber(inputfile)

    record = '{0} {1} {2} {3}'.format(seqno,
                                      os.path.abspath(inputfile),
                                      rpreddtypes.genhash(args.centre,
                                                          args.sensitive,
                                                          args.heavy),
                                      args.rotations)

    truevals = rainPresent(rpReader, args.sensitive, args.bounds, args.heavy,
                           seqno)
    record = '{0} {1} {2}'.format(record, truevals[0], truevals[1])

    for rot in range(args.rotations):
        sense2 = args.sensitive.copy()
        for i in range(len(sense2)):
            sense2[i] = rotate_pixel_CCW(sense2[i], args.centre,
                                         args.rotations + 1, rot + 1)
        truevals = rainPresent(rpReader, sense2, args.bounds, args.heavy)
        record = '{0} {1} {2}'.format(record, truevals[0], truevals[1])

    print (record)
        
