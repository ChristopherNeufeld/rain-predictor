#! /usr/bin/python3

import rpreddtypes
import argparse
import numpy
import gif
import math
import sys


numRings = 20
numRadialCuts = 20


numModules = numRings * numRadialCuts
modules = numpy.full((480, 480), -1)

greyscale = []
for i in range(256):
    greyscale.append([i, i, i])

# Modules are numbered clockwise (because this is a
# left-handded coordinate system) from the X axis, closest
# ring first.

radius = int(480 / 2)
r2 = radius ** 2
for pixelCol in range(2 * radius):
    for pixelRow in range(2 * radius):
        delta = [ pixelCol - radius, pixelRow - radius ]
        d2 = delta[0] ** 2 + delta[1] ** 2
        if d2 > r2:
            continue

        ringnum = int(math.sqrt(d2) / radius * numRings)
        if ringnum >= numRings:
            continue  # shouldn't happen for reasonable width floats

        angle = math.atan2(delta[1], delta[0])
        if angle < 0:
            angle += 2 * math.pi

        secnum = int(angle / (2 * math.pi) * numRadialCuts)

        # More "shouldn't happen" floating point defense
        if secnum < 0:
            secnum = 0
        if secnum >= numRadialCuts:
            secnum = numRadialCuts - 1

        modnum = int(ringnum * numRadialCuts + secnum)
        modules[pixelRow, pixelCol] = modnum

fileind = 0
for arg in sys.argv[1:]:
    reader = rpreddtypes.RpBinReader()
    reader.read(arg)
    prepobj = reader.getPreparedDataObject()
    prepdata = prepobj.getPreparedData()
    numRows = reader.getHeight()
    numCols = reader.getWidth()

    output1 = numpy.zeros([numRows, numCols], dtype=numpy.uint8)
    output2 = numpy.zeros([numRows, numCols], dtype=numpy.uint8)
    output3 = numpy.zeros([numRows, numCols], dtype=numpy.uint8)

    for row in range(numRows):
        for col in range(numCols):
            modnum = modules[row, col]
            if modnum != -1:

                output1[row, col] = prepdata[modnum * 3]
                output2[row, col] = prepdata[modnum * 3 + 1]
                output3[row, col] = prepdata[modnum * 3 + 2]

                if ( prepdata[modnum * 3] < 0 or
                     prepdata[modnum * 3 + 1] < 0 or
                     prepdata[modnum * 3 + 2] < 0) :
                    print('Bad data for module {}:  {} {} {}'.format(modnum, prepdata[modnum * 3], prepdata[modnum * 3 + 1], prepdata[modnum * 3 + 2]))
                


    oputs = [ output1, output2, output3 ]
    names = [ 'file-{}-1.gif'.format(fileind),
              'file-{}-2.gif'.format(fileind),
              'file-{}-3.gif'.format(fileind)]
    fileind = fileind + 1

    for contents, name in zip(oputs, names):
        owidth = numCols
        oheight = numRows
        odepth = 8
        ocolours = greyscale
        oblock = []

        for row in range(numRows):
            for col in range(numCols):
                if contents[row, col] < 0:
                    print('Bad contents {}/{} at {},{}'.format(contents[row, col], name, row, col))
                oblock.append(contents[row, col])

        writer = gif.Writer(open(name, 'wb'))
        writer.write_header()
        writer.write_screen_descriptor(owidth, oheight, True, odepth)
        writer.write_color_table(ocolours, odepth)
        writer.write_image(owidth, oheight, odepth, oblock)
        writer.write_trailer()
    
