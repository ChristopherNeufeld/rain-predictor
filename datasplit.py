#! /usr/bin/python3

# Split the candidates file into training and validation sets.

import argparse
import random


parser = argparse.ArgumentParser(description='Split data into '
                                 'training and validation sets.')
parser.add_argument('--candidates', type=str, dest='inputfile',
                    required=True,
                    help='Path of the candidates file produced '
                    'by get-training-set.py')
parser.add_argument('--veto-set', type=str, dest='vetoset',
                    help='If supplied, its argument is the name '
                    'of a file containing candidates that '
                    'are to be skipped.')
parser.add_argument('--training-file', type=str, dest='trainingFile',
                    required=True,
                    help='The pathname into which training data '
                    'will be written.')
parser.add_argument('--validation-file', type=str, dest='validationFile',
                    required=True,
                    help='The pathname into which validation data '
                    'will be written.')
parser.add_argument('--validation-fraction', type=float,
                    dest='validationFrac', default=0.2,
                    help='The fraction of candidates that will be '
                    'reserved for validation.')
parser.add_argument('--validation-count', type=int,
                    dest='validationCount', default=0,
                    help='The number of candidates that will be '
                    'reserved for validation.  '
                    'Supersedes --validation-fraction.')

args = parser.parse_args()


vetoes = []

if args.vetoset:
    with open(args.vetoset, 'r') as ifile:
        for record in ifile:
            fields = record.split()
            vetolist.append(int(fields[0]))

validcandidates = []
trainingset = []
validationset = []

with open(args.inputfile, 'r') as ifile:
    for record in ifile:
        fields = record.split()
        if int(fields[0]) in vetoes:
            continue
        validcandidates.append(record)

random.shuffle(validcandidates)
numValid = len(validcandidates)
numReserved = 0

if args.validationCount == 0:
    numReserved = int(args.validationFrac * numValid)
else:
    numReserved = args.validationCount

with open(args.trainingFile, 'w') as ofile:
    trainingset = validcandidates[numReserved:]
    trainingset.sort()
    for line in trainingset:
        ofile.write('{}'.format(line))

with open(args.validationFile, 'w') as ofile:
    validationset = validcandidates[:numReserved]
    validationset.sort()
    for line in validationset:
        ofile.write('{}'.format(line))
    
