#! /usr/bin/python3

# Uses pre-trained networks to make predictions.


import rpreddtypes
import argparse
import sys
import numpy
import keras


def histBinNum(val, nBins):
    if val >= 1:
        return nBins - 1
    if val <= 0:
        return 0
    return int(nBins * val)


# Main execution begins here


suffixes = [ '1H_R', '1H_HR', '2H_R', '2H_HR', '3H_R', '3H_HR',
             '4H_R', '4H_HR', '5H_R', '5H_HR' ]
             



parser = argparse.ArgumentParser(description='Make rain prediction(s).')
parser.add_argument('--pathfile', type=str, dest='pathfile',
                    required = True,
                    help='The file that maps sequence numbers to '
                    'the pathnames of the binary files.')
parser.add_argument('--testdata', type=str, dest='candidates',
                    help='A candidates-style file with predictions.')
parser.add_argument('--histograms', type=bool, dest='withHist',
                    default = False,
                    help='If set, will write data suitable for '
                    'histogramming into several files.  Filenames are '
                    'build on a prefix of "rphist-" unless overridden.')
parser.add_argument('--num-bins', type=int, dest='histBins',
                    default = 20,
                    help='The number of bins for histogramming.')
parser.add_argument('--histogram-prefix', type=str, dest='histPrefix',
                    default = 'rphist-',
                    help='Override filename prefix for writing '
                    'histogram files.')
parser.add_argument('--one-shot', type=int, dest='singleTest',
                    help='A sequence number for which to produce a '
                    'prediction.')
parser.add_argument('--saved-network', type=str, dest='savefile',
                    required = True,
                    help='The filename holding the complete saved '
                    'network (not just the weights).')

args = parser.parse_args()

if not args.pathfile:
    print('A pathfile is always required to connect sequence numbers '
          'to intermediate binary file pathnames.')
    sys.exit(1)

if args.candidates and args.singleTest:
    print('Only one of --testdata and --one-shot should be supplied')
    sys.exit(1)

if args.singleTest and args.withHist:
    print('It doesn\'t make sense to make a histogram of a single '
          'datapoint.')
    sys.exit(1)


    

if args.singleTest:
    print('Not yet implemented')
    sys.exit(1)


mymodel = keras.models.load_model(args.savefile)

hdata = None


# We bin columns:  True negative, false negative, false positive, true positive
if args.withHist:
    hdata = numpy.zeros([10, 4, args.histBins])

xvals = None
yvals = None
datasize = None
npts = None
hashval = None
    
if args.candidates:
    xvals, yvals, datasize, npts, hashval = rpreddtypes.getDataVectors(args.candidates, args.pathfile, doShuffle = False)

    ypred = mymodel.predict(x = xvals)

    if args.withHist:
        for datapt in range(npts):
            for bitnum in range(10):
                if yvals[datapt, bitnum] == 0:
                    if ypred[datapt, bitnum] < 0.5:
                        binnum = histBinNum(ypred[datapt, bitnum],
                                            args.histBins)
                        hdata[bitnum, 0, binnum] += 1
                    else:
                        binnum = histBinNum(ypred[datapt, bitnum],
                                            args.histBins)
                        hdata[bitnum, 2, binnum] += 1
                else:
                    if ypred[datapt, bitnum] < 0.5:
                        binnum = histBinNum(ypred[datapt, bitnum],
                                            args.histBins)
                        hdata[bitnum, 1, binnum] += 1
                    else:
                        binnum = histBinNum(ypred[datapt, bitnum],
                                            args.histBins)
                        hdata[bitnum, 3, binnum] += 1

        for bitnum in range(10):
            with open(args.histPrefix + suffixes[bitnum], 'w') as ofile:
                ofile.write('# Bin_Centre    TN   FN   FP   TP\n')
                for binnum in range(args.histBins):
                    ofile.write('{0}  {1}  {2}  {3}  {4}\n'
                                .format((binnum + 0.5) / args.histBins,
                                        hdata[bitnum, 0, binnum],
                                        hdata[bitnum, 1, binnum],
                                        hdata[bitnum, 2, binnum],
                                        hdata[bitnum, 3, binnum]))
                        
