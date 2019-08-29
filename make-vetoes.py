#! /usr/bin/python3

# Before we go off cutting out candidates from the training set, we
# should make sure we understand the data we're eliminating.  So,
# we'll generate a data set that can be used to plot candidate rain
# intensity on the Y axis and number of training candidates with
# values at or below that sum on the X axis

import argparse
import rpreddtypes
import sys

parser = argparse.ArgumentParser(description='Generate '
                                 'intensity data for plotting.')

parser.add_argument('--candidates', type=str,
                    dest='candidates',
                    help='A file of candidates data from '
                    'get-training-set.py')

parser.add_argument('--sequences', type=str,
                    dest='sequences',
                    help='A file of sequence information from '
                    'prepare-true-vals.py')

parser.add_argument('--with-plotting-data', type=bool,
                    dest='plotdat', default=True,
                    help='Whether to generate plotting data '
                    'on stdout')
parser.add_argument('--veto-fraction', type=float,
                    dest='vetofrac', default=0.5,
                    help='Fraction of no-rain candidates to put in '
                    'the veto list for exclusion.')
parser.add_argument('--veto-keepstride', type=int,
                    dest='keepstride', default=6,
                    help='When writing vetoes, will exclude each '
                    'Nth entry from the list to ensure we don\'t '
                    'produce data that\'s too lopsided.')
parser.add_argument('--write-thinned-data', type=str,
                    dest='thinnedOutput',
                    help='Pass a filename into which the thinned '
                    'data will be written, the vetoes having '
                    'been applied.')
parser.add_argument('--write-vetoes', type=str,
                    dest='vetofile',
                    help='If set, writes an unsorted list of '
                    'candidate sequence '
                    'numbers to skip because of common-case thinning.')

args = parser.parse_args()

if not args.candidates:
    print('A file with the list of candidates must be supplied with the '
          '--candidates argument')
    sys.exit(1)

if not args.sequences:
    print('A file with the list of sequence data be supplied with the '
          '--sequences argument')
    sys.exit(1)

seqIntensity = {}
sumIntensities = []
with open(args.sequences, 'r') as ifile:
    for record in ifile:
        fields = record.split()
        seqno = int(fields[0])
        pathname = fields[1]
        reader = rpreddtypes.RpBinReader()
        reader.readHeader(pathname)
        seqIntensity[seqno] = reader.getTotalRain()
    
with open(args.candidates, 'r') as ifile:
    for record in ifile:
        fields = record.split()
        startval = int(fields[0])

        skipEntry = False
        for i in range(4,14):
            if (fields[i] != '0'):
                skipEntry = True
                break

        if skipEntry:
            continue
        
        sum = 0
        for i in range(6):
            sum += seqIntensity[startval + i]

        sumIntensities.append([sum, startval])

sumIntensities.sort()

if args.plotdat:
    for i in range(len(sumIntensities)):
        print (sumIntensities[i][0], i)


vetoes=[]
recordsToDiscard = int(len(sumIntensities) * args.vetofrac)
for i in range(len(sumIntensities)):
    if recordsToDiscard > 0:
        if i % args.keepstride != 0:
            vetoes.append(sumIntensities[i][1])
            recordsToDiscard -= 1
    else:
        break
        
if args.vetofile:
    with open(args.vetofile, 'w') as ofile:
        for v in vetoes:
            ofile.write('{}\n'.format(v))

if args.thinnedOutput:
    with open(args.candidates, 'r') as ifile:
        with open(args.thinnedOutput, "w") as ofile:
            for record in ifile:
                fields = record.split()
                seqno = int(fields[0])
                if seqno in vetoes:
                    continue
                ofile.write(record)
