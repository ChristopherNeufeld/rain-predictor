#! /usr/bin/python3
#
#
# The data from the archive includes 4 optional overlays that are not
# obtained from the up-to-date radar site.  This code reads in those
# overlays and produces an overlay .gif file.  Pixels will be black if
# they occlude rain pixels, and white if rain pixels are visible
# there.

import argparse
import gif
import sys

parser = argparse.ArgumentParser(description='Make overlay .gif file')
parser.add_argument('ifilenames', type=str,
                    metavar='filename', nargs='+',
                    help='Filenames to process')
parser.add_argument('--output', type=str,
                    dest='output',
                    help='The resulting output .gif file')

args = parser.parse_args()

if not args.output:
    print('An output file name is required to hold the result.')
    sys.exit(1)

output_block = []
width = -1
height = -1

for ol in args.ifilenames:

    transparentColour = None
    gifdata = None
    
    greader = gif.Reader()
    gfile = open(ol, 'rb')
    greader.feed(gfile.read())
    gfile.close()

    if width == -1:
        width = greader.width

    if height == -1:
        height = greader.height

    for block in greader.blocks:
        if isinstance(block, gif.GraphicControlExtension):
            transparentColour = block.transparent_color

        if isinstance(block, gif.Image):
            gifdata = block.get_pixels()

    if not transparentColour:
        print('Failed to find transparent colour of overlay')
        sys.exit(1)

    if not gifdata:
        print('Failed to find pixel content overlay')
        sys.exit(1)
        
    npixels = len(gifdata)

    if len(output_block) == 0:
        output_block = [1] * npixels

    for i in range(npixels):
        if gifdata[i] != transparentColour:
            output_block[i] = 0
    


writer = gif.Writer(open(args.output, 'wb'))
writer.write_header()
writer.write_screen_descriptor(width, height, True, 1)
writer.write_color_table([(0, 0, 0), (255, 255, 255)], 1)
writer.write_image(width, height, 1, output_block)
writer.write_trailer()

