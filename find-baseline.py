#! /usr/bin/python3

# This script reads in three .gif files and produces a new file in
# which each pixel is set to the majority value from the three inputs.
# If there is no majority value (i.e. all three files have a different
# value at that point), we exit with an error so that a better set of
# inputs can be found.

# We are using this script to analyse machine-generated files in a
# single context.  While the usual programming recommendation is to be
# very permissive in what formats you accept, I'm going to restrict
# myself to verifying consistency and detecting unexpected inputs,
# rather than trying to handle all of the possible cases.

# This is a pre-processing step that will be used by another script
# that reads .gif files.  Therefore it is reasonable to make this
# script's output be a .gif itself.

# The script takes 4 arguments.  The first three are the names of the
# input files.  The fourth is the name of the output file.

# The script will return '1' on error, '0' for success.

import sys
import gif


class SearchFailed(Exception):
    def __init__(self, message):
        self.message = message


def find_index_of_tuple (list_of_tuples, needle, hint = 0):
    if list_of_tuples[hint] == needle:
        return hint
    for i in list_of_tuples:
        if (list_of_tuples[i] == needle):
            return i
    raise SearchFailed('Tuple {0} not found in list.' % needle)


if len(sys.argv) != 5:
    print ("Require 3 input filenames and 1 output filename.")
    sys.exit(1)

file = [None, None, None]
reader = [None, None, None]

for i in range(3):    
    try:    
        file[i] = open(sys.argv[i+1], 'rb')
    except OSError as ex:
        print ("Failed to open input file: ", sys.argv[i+1])
        print ("Reason: ", ex.strerror)
        sys.exit(1)
    reader[i] = gif.Reader()
    reader[i].feed(file[i].read())
    if ( not reader[i].is_complete()
         or not reader[i].has_screen_descriptor() ):
        print ("Failed to parse", sys.argv[i+1], "as a .gif file")
        sys.exit(1)

# OK, if we get here it means we have successfully loaded three .gif
# files.  The user might have handed us the same one three times, but
# there's not much I can do about that, it's entirely possible that we
# want to look at three identical but distinct files, and filename
# aliases make any more careful examination of the paths platform
# dependent.

# So, we're going to want to verify that the three files have the same
# sizes.

if ( reader[0].width != reader[1].width
     or reader[1].width != reader[2].width
     or reader[0].height != reader[1].height
     or reader[1].height != reader[2].height ):
    print ("The gif logical screen sizes are not identical")
    sys.exit(1)


imgoffsets = [ 0 ] * 3    

for i in range(3):

    valid = False

    if ( len(reader[i].blocks) == 2
         and isinstance(reader[i].blocks[0], gif.Image)
         and isinstance(reader[i].blocks[1], gif.Trailer)):

        valid = True
        imgoffsets[i] = 0


    if ( len(reader[i].blocks) == 3
         and isinstance(reader[i].blocks[0], gif.GraphicControlExtension)
         and isinstance(reader[i].blocks[1], gif.Image)
         and isinstance(reader[i].blocks[2], gif.Trailer)):

        valid = True
        imgoffsets[i] = 1


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
    
    
# Time to vote

try:
    writer = gif.Writer (open (sys.argv[4], 'wb'))
except OSError as ex:
    print ("Failed to open output file: ", sys.argv[4])
    print ("Reason: ", ex.strerror)
    sys.exit(1)

output_width = reader[0].width
output_height = reader[0].height
output_colour_depth = 8
output_colour_table = reader[0].color_table
output_pixel_block = []


for ind0, ind1, ind2 in zip(reader[0].blocks[imgoffsets[0]].get_pixels(),
                            reader[1].blocks[imgoffsets[1]].get_pixels(),
                            reader[2].blocks[imgoffsets[2]].get_pixels()):
    tup0 = reader[0].color_table[ind0]
    tup1 = reader[1].color_table[ind1]
    tup2 = reader[2].color_table[ind2]

    # Voting
    if ( tup0 == tup1 or tup0 == tup2):
        output_pixel_block.append(ind0)
    elif ( tup1 == tup2 ):
        try:
            newind = find_index_of_tuple(output_colour_table,
                                         tup1, ind1)
            output_pixel_block.append(newind)
        except SearchFailed as ex:
            print ('The colour table for file %s does not hold the '
                   'entry {0} that won the vote.  You may be able '
                   'to fix this problem simply by reordering your '
                   'command-line arguments.' % sys.argv[1], tup1)
            sys.exit(1)

writer.write_header()
writer.write_screen_descriptor(output_width, output_height,
                               True, output_colour_depth)
writer.write_color_table(output_colour_table, output_colour_depth)
writer.write_image(output_width, output_height,
                   output_colour_depth, output_pixel_block)
writer.write_trailer()
