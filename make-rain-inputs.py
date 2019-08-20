#! /usr/bin/python3

# This script will take .gif files downloaded from the radar station
# and convert them to a simpler format for eventual use.  It will
# contain only information about precipitation or its absence, in a
# set of integer steps.

import argparse
import sys
import gif
import rpreddtypes

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
    
for ifile in args.ifilenames:
    convertReader = gif.Reader()
    cfile = open(ifile, 'rb')
    convertReader.feed(cfile.read())
    cfile.close()

    totalRain = 0
    
    if ( not convertReader.is_complete()
         or not convertReader.has_screen_descriptor() ):
        print ('Failed to parse {0} as a '
               '.gif file'.format(ifile))
        sys.exit(1)

    if ( len(convertReader.blocks) != 2
         or not isinstance(convertReader.blocks[0], gif.Image)
         or not isinstance(convertReader.blocks[1], gif.Trailer)):
        print ('While processing file: {}'.format(ifile))
        print ('The code only accepts input files with a single block of '
               'type Image followed by one of type Trailer.  This '
               'constraint has not been met, the code will have to be '
               'changed to handle the more complicated case.')
        sys.exit(1)
        
    convertBuffer = convertReader.blocks[0].get_pixels()
    convertColours = convertReader.color_table
    convertWidth = convertReader.width
    convertHeight = convertReader.height

    if baselineWidth != convertWidth or baselineHeight != convertHeight:
        print('The baseline file ({0}) and the file to convert {1} '
              'have incompatible dimensions'.format(args.baseline,
                                                    ifile))
        sys.exit(1)

    output_block = []

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

    newfilename = ifile + '.bin'

    writer = rpreddtypes.RpBinWriter()
    writer.write(newfilename, newwidth, newheight, xoffset, yoffset,
                 totalRain, output_block)

    if (args.verbose):
        print('Wrote output file: {0}'.format(newfilename))

