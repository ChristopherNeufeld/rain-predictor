#! /usr/bin/python3

# This script will take .gif files downloaded from the radar station
# and convert them to a simpler format for eventual use.  It will
# contain only information about precipitation or its absence, in a
# set of integer steps.

import argparse
import sys
import gif
import rpreddtypes
from rpreddtypes import normalize
import numpy
import math




### Main entry point starts here



parser = argparse.ArgumentParser(description='Extract '
                                 'precipitation data.')
parser.add_argument('ifilenames', type=str,
                    metavar='filename', nargs='+',
                    help='Filenames to process')
parser.add_argument('--baseline', type=str,
                    dest='baseline',
                    help='The baseline .gif file')
parser.add_argument('--width', type=int, dest='owidth',
                    default=-1,
                    help='The width of the sub-rectangle '
                    'that is to be output')
parser.add_argument('--height', type=int, dest='oheight',
                    default=-1,
                    help='The height of the sub-rectangle '
                    'that is to be output')
parser.add_argument('--top-left-x', type=int, dest='offsetx',
                    default=0,
                    help='The x-value of the upper left '
                    'of the sub-rectangle that is to be output')
parser.add_argument('--top-left-y', type=int, dest='offsety',
                    default=0,
                    help='The y-value of the upper left '
                    'of the sub-rectangle that is to be output')
parser.add_argument('--override-intensities', type=list,
                    dest='intensities',
                    default=[0x99ccff, 0x0099ff, 0x00ff66,
                             0x00cc00, 0x009900, 0x006600,
                             0xffff33, 0xffcc00, 0xff9900,
                             0xff6600, 0xff0000, 0xff0299,
                             0x9933cc, 0x660099],
                    help='Override the colour codes for '
                    'intensities')
parser.add_argument('--override-scaling', type=list,
                    dest='rescales', default=[],
                    help='Override the coarse-scaling settings.')
parser.add_argument('--build-preprocessed', type=bool, dest='preproc',
                    default = True,
                    help = 'Build preprocessed full-resolution '
                    'modules (recommended)')
parser.add_argument('--preprocessed-num-rings', type=int, dest='numRings',
                    default = 20,
                    help = 'Number of rings of modules to produce '
                    'when preprocessing.')
parser.add_argument('--preprocessed-num-cuts', type=int,
                    dest='numRadialCuts',
                    default = 20,
                    help = 'Number of radial cuts to produce '
                    'when preprocessing.')
parser.add_argument('--heavy', type=int, dest='heavy',
                    default = 3,
                    help = 'Intensity of heavy rain, for use when '
                    'producing pre--processed inputs.')
parser.add_argument('--verbose', type=bool, dest='verbose',
                    default = False,
                    help='Extra output during processing')

args = parser.parse_args()

if not args.baseline:
    print ('A baseline comparison file must be supplied '
           'with the --baseline argument')
    sys.exit(1)

baselineReader = gif.Reader()
bfile = open(args.baseline, 'rb')
baselineReader.feed(bfile.read())
bfile.close()
if ( not baselineReader.is_complete()
     or not baselineReader.has_screen_descriptor() ):
    print ('Failed to parse {0} as a '
           '.gif file'.format(args.baseline))
    sys.exit(1)

baselineBuffer = baselineReader.blocks[0].get_pixels()
baselineColours = baselineReader.color_table
baselineWidth = baselineReader.width
baselineHeight = baselineReader.height

newwidth = baselineWidth
if args.owidth != -1:
    newwidth = args.owidth

newheight = baselineHeight
if args.oheight != -1:
    newheight = args.oheight

xoffset = args.offsetx
yoffset = args.offsety
modules = None
numModules = 0

if args.preproc:
    # For each module, we store a list of pixels.  We walk the
    # pixel space and assign each one to a single module.  This
    # does not, in general, guarantee that the modules have
    # exactly the same number of pixels, but that's not an
    # important consideration here.

    numModules = args.numRings * args.numRadialCuts
    modules = numpy.full((newheight, newwidth), -1)

    # Modules are numbered clockwise (because this is a
    # left-handded coordinate system) from the X axis, closest
    # ring first.

    radius = int(newwidth / 2)
    r2 = radius ** 2
    for pixelCol in range(2 * radius):
        for pixelRow in range(2 * radius):
            delta = [ pixelCol - radius, pixelRow - radius ]
            d2 = delta[0] ** 2 + delta[1] ** 2
            if d2 > r2:
                continue

            ringnum = int(math.sqrt(d2) / radius * args.numRings)
            if ringnum >= args.numRings:
                continue  # shouldn't happen for reasonable width floats

            angle = math.atan2(delta[1], delta[0])
            if angle < 0:
                angle += 2 * math.pi

            secnum = int(angle / (2 * math.pi) * args.numRadialCuts)

            # More "shouldn't happen" floating point defense
            if secnum < 0:
                secnum = 0
            if secnum >= args.numRadialCuts:
                secnum = args.numRadialCuts - 1

            modnum = int(ringnum * args.numRadialCuts + secnum)
            modules[pixelRow, pixelCol] = modnum


for ifile in args.ifilenames:
    convertReader = gif.Reader()
    cfile = open(ifile, 'rb')
    convertReader.feed(cfile.read())
    cfile.close()

    totalRain = 0
    imgoffset = 0
    valid = False
    
    if ( not convertReader.is_complete()
         or not convertReader.has_screen_descriptor() ):
        print ('Failed to parse {0} as a '
               '.gif file'.format(ifile))
        sys.exit(1)

    if ( len(convertReader.blocks) == 2
         and isinstance(convertReader.blocks[0], gif.Image)
         and isinstance(convertReader.blocks[1], gif.Trailer)):

        valid = True
        imgoffset = 0

    if ( len(convertReader.blocks) == 3
         and isinstance(convertReader.blocks[0], gif.GraphicControlExtension)
         and isinstance(convertReader.blocks[1], gif.Image)
         and isinstance(convertReader.blocks[2], gif.Trailer)):

        valid = True
        imgoffset = 1
    

    if not valid:
        print ("While processing file: ", sys.argv[i+1])
        print ("The code only accepts input files with a single block of "
               "type Image followed by one of type Trailer, and optionally "
               "a graphic control extension block before the image block.  "
               "This "
               "constraint has not been met, the code will have to be "
               "changed to handle the more complicated case.")

        print('blocks: {}'.format(reader[i].blocks))
        sys.exit(1)

        
    convertBuffer = convertReader.blocks[imgoffset].get_pixels()
    convertColours = convertReader.color_table
    convertWidth = convertReader.width
    convertHeight = convertReader.height

    if baselineWidth != convertWidth or baselineHeight != convertHeight:
        print('The baseline file ({0}) and the file to convert {1} '
              'have incompatible dimensions'.format(args.baseline,
                                                    ifile))
        sys.exit(1)

    output_block = []
    preprocessed = None
    counts = None
    
    if args.preproc:
        preprocessed = bytearray(2 * numModules)
        counts = [0] * numModules
        sums = [0] * numModules

    for pixel in range(len(baselineBuffer)):

        row = pixel // baselineWidth
        col = pixel % baselineWidth

        if row < yoffset:
            continue

        if row >= yoffset + newheight:
            break

        if col < xoffset or col >= xoffset + newwidth:
            continue

        if pixel >= len(convertBuffer):
            output_block.append(0)
            continue

        btuple = baselineColours[baselineBuffer[pixel]]
        ctuple = convertColours[convertBuffer[pixel]]
        
        if btuple == ctuple:
            output_block.append(0)
        else:
            code = ( ctuple[0] * 256 * 256
                     + ctuple[1] * 256
                     + ctuple[2] )
            appendval = 0
            for i in range(len(args.intensities)):
                if code == args.intensities[i]:
                    appendval = i+1
                    break
            output_block.append(appendval)
            totalRain += appendval

            if args.preproc and appendval != 0:
                modnum = modules[col][row]
                sums[modnum] += appendval
                counts[modnum] += 1
                if appendval > preprocessed[2 * modnum]:
                    preprocessed[2 * modnum] = appendval

    newfilename = ifile + '.bin'

    if args.preproc:
        for modnum in range(numModules):
            if counts[modnum] > 0:
                preprocessed[2 * modnum] = int (normalize(preprocessed[2 * modnum],
                                                          args.heavy,
                                                          len(args.intensities),
                                                          5)
                                                * 255)
                preprocessed[2 * modnum + 1] = int (normalize(sums[modnum] / counts[modnum],
                                                              args.heavy,
                                                              len(args.intensities),
                                                              5)
                                                    * 255)

    writer = rpreddtypes.RpBinWriter()
    writer.addRawdat(newwidth, newheight, xoffset, yoffset,
                     len(args.intensities), output_block, [2, 3, 4])
    writer.addPreparedData(2 * numModules, args.numRings,
                           args.numRadialCuts, preprocessed)
    writer.write(newfilename, len(args.intensities), totalRain)

    if (args.verbose):
        print('Wrote output file: {0}'.format(newfilename))

