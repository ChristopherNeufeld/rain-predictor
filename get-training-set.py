#! /usr/bin/python3

# Here we generate the training set candidates.
#
# We read the list of records made by prepare-true-vals.py from a
# file, and produce a list of candidates on stdout
#
# A training set candidate is a run of 6 sequence numbers representing
# the previous hour of historical data, while the following 30
# sequence numbers inform the rain/no-rain data for the five 1-hour
# blocks of future predictions.
#
# A training set candidate is available when there exists a set of 36
# consecutive sequence numbers.  In that case, we generate a record
# like this:
#
# <FIRST_SEQ_NO> <HASH> <IS_RAINING> <N_ROTS> <ROTNUM> <RAIN0_1> <HEAVY0_1> ...
#
# The first field is the starting sequence number in the run of 36.
#
# The second is the hash, as in prepare-true-vals.py, to ensure that
# we don't accidentally mix incompatible training data
#
# The third field is a boolean value that indicates whether the last
# timestep in the historical record is showing rain.  That way we can
# see how well the network predicts transitions, instead of having it
# just tell us that the rain/not rain continues.

# The fourth field is the number of rotations from which this is taken,
# or 0 if we're using unrotated data sets
#
# The fifth field is the rotation index.  0 for unrotated, up to 1
# less than N_ROTS
#
# The sixth field is the logical OR of the RAIN record for the 7th
# through 12th sequence numbers.  The sixth is the logical OR of the
# HEAVY_RAIN record for the 7th through 12th sequence numbers.
#
# The following 8 fields are as above, for subsequence runs of 6
# sequence numbers.

import argparse
import rpreddtypes

parser = argparse.ArgumentParser(description='Find training candidates.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('truevalfile', type=str, metavar='truevalfile',
                    help='Filename to process')

parser.add_argument('--override-centre', type=list, dest='centre',
                    default=[240,239], help='Set a new location for '
                    'the pixel coordinates of the radar station')
parser.add_argument('--override-sensitive-region', type=list,
                    dest='sensitive',
                    default=[[264,204], [264,205], [265,204], [265,205]],
                    help='Set a new list of sensitive pixels')
parser.add_argument('--rotations', type=int, dest='rotations',
                    default=0, help='Number of synthetic data points '
                    'to create (via rotation) for each input data point')
parser.add_argument('--heavy-rain-index', type=int, dest='heavy',
                    default=3, help='Lowest index in the colour table '
                    'that indicates heavy rain, where 1 is the '
                    'lightest rain.')

args = parser.parse_args()

hashval = rpreddtypes.genhash(args.centre, args.sensitive, args.heavy)

seqnoList = []
parsedData = {}

nRots = -1
skipRotations = False

with open(args.truevalfile, 'r') as ifile:
    for record in ifile:
        fields = record.split()
        if fields[2] != hashval:
            continue

        # Check for inconsistent number of rotations
        if nRots != -1 and nRots != int(fields[3]):
            skipRotations = True
            
        nRots = int(fields[3])
        seqnoList.append(int(fields[0]))
        value = [ int(fields[3]) ]
        value[1:1] = list(map(int, fields[4:]))
        parsedData[int(fields[0])] = value

seqnoList.sort()
if skipRotations:
    nRots = 0

# Now, we've loaded sequence numbers into a list, and indexed a dict
# against them to record number of rotations and rain data.  

# Find runs of 36.

idx = 0
while idx < len(seqnoList) - 36:
    candSeqNo = seqnoList[idx]
    
    offset = 1
    invalid = False
    while not invalid and offset < 36:
        if seqnoList[idx + offset] != seqnoList[idx + offset - 1] + 1:
            invalid = True
            idx = idx + offset

        offset += 1

    if invalid:
        continue

    for rot in range(nRots + 1):
        lastHistoricalRecord = parsedData[candSeqNo + 5]
        rainingNow = lastHistoricalRecord[1 + rot * 2]
        record = '{0} {1} {2} {3} {4} '.format(candSeqNo, hashval,
                                               rainingNow, nRots, rot)
        binnedValues = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        for timeInterval in range(5):
            for snapshot in range(6):
                oneRec = parsedData[candSeqNo + 6 + timeInterval * 6 + snapshot]
                if oneRec[1 + rot * 2] == 1:
                    binnedValues[timeInterval * 2] = 1
                if oneRec[1 + rot * 2 + 1] == 1:
                    binnedValues[timeInterval * 2 + 1] = 1

        print(record, *binnedValues, sep = ' ')

    idx += 1
    
