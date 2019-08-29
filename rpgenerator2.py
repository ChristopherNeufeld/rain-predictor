#! /usr/bin/python3


# THIS FILE IS NOW OBSOLETE.  FUNCTIONALITY ENTIRELY SUBSUMED INTO
# rptrainer2.py


# The data generator for the rain predictor
import keras
# from tensorflow import keras
import rpreddtypes
import math
import numpy
import random


class RPDataGenerator2(keras.utils.Sequence):
    'Produces data from the training set records we\'ve built'

    # Magic behaviour, if batch_size is fractional, we make batches of
    # size equal to that fraction of the total training set
    def __init__(self, sequence_file, path_file, veto_file,
                 centre, sensitive_region, heavyThreshold,
                 batch_size):
        self.ringcount = 20
        self.radialcuts = 40
        self.sequence_file = sequence_file
        self.path_file = path_file
        self.veto_file = veto_file
        self.centre = centre
        self.sensitive_region = sensitive_region
        self.heavy = heavyThreshold
        self.datasize = None
        self.num_intensities = None
        self.intensity_gap = 5    # to separate light from heavy rain
                                  # in the inputs
        self.batch_size = batch_size
        self.hash = rpreddtypes.genhash(centre, sensitive_region,
                                        heavyThreshold)
        self.pathmap = {}
        self.seqmap = {}
        self.seqlist = []

        self.useCachedData = False
        self.fullsetX = None
        self.fullsetY = None

        self.loadFromFiles()

    def loadFromFiles(self):
        with open(self.path_file, 'r') as ifile:
            for record in ifile:
                fields = record.split()
                seqno = int(fields[0])
                self.pathmap[seqno] = fields[1]

        vetolist = []
        if self.veto_file:
            with open(self.veto_file, 'r') as ifile:
                for record in ifile:
                    fields = record.split()
                    seqno = int(fields[0])
                    vetolist.append(seqno)
            
        with open(self.sequence_file, 'r') as ifile:
            for record in ifile:
                fields = record.split()
                seqno = int(fields[0])
                hashno = fields[1]
                if hashno != self.hash:
                    continue
                if seqno in vetolist:
                    continue
                self.seqmap[seqno] = list(map(int, fields[4:]))
                self.seqlist.append(seqno)

        seqno = self.seqlist[0]
        filename = self.pathmap[seqno]
        inputs = self.inputsFromOneFile(filename)
        if self.batch_size == 0:
            self.batch_size = len(self.seqlist)
            self.cacheEntireInputSet()

        if self.batch_size > 0 and self.batch_size < 1:
            self.batch_size = int(self.batch_size * len(self.seqlist))

        if self.batch_size < len(self.seqlist):
            self.shuffleSequence()


    def shuffleSequence(self):
        random.shuffle(self.seqlist)

    def on_epoch_end(self):
        if self.batch_size < len(self.seqlist):
            self.shuffleSequence()

    def getBatchSize(self):
        return self.batch_size

    def __len__(self):
        return len(self.seqlist) // self.batch_size

    def getInputSize(self):
        return self.datasize

    def inputsFromOneFile(self, filename):
        reader = rpreddtypes.RpBinReader()
        reader.read(filename)
        if not self.num_intensities:
            self.num_intensities = reader.getMaxRainval()

        rpbo = reader.getPreparedDataObject()
        if not self.datasize:
            self.datasize = rpbo.getDataLength()
        return numpy.asarray(rpbo.getPreparedData()) / 255


    def cacheEntireInputSet(self):
        self.useCachedData = True
        self.shuffle = False
        self.fullsetX = numpy.empty([self.batch_size, 6, self.datasize])
        self.fullsetY = numpy.empty([self.batch_size, 10])

        for oib in range(self.batch_size):
            base_seqno = self.seqlist[oib]
            for ts in range(6):
                seqno = base_seqno + ts
                filename = self.pathmap[seqno]
                inputs = self.inputsFromOneFile(filename)
                self.fullsetX[oib][ts] = inputs

            self.fullsetY[oib] = numpy.asarray(self.seqmap[base_seqno])
        
        
    def __getitem__(self, index):
        'Return one batch'
        # Return xvals in a numpy array
        # indexed by [offsetInBatch][timestep][valindex]
        # Hand the y-vals back in another array

        if self.useCachedData:
            return self.fullsetX, self.fullsetY

        rvalX = numpy.empty([self.batch_size, 6, self.datasize])
        rvalY = numpy.empty([self.batch_size, 10])

        for oib in range(self.batch_size):
            base_seqno = self.seqlist[index * self.batch_size + oib]
            for ts in range(6):
                seqno = base_seqno + ts
                filename = self.pathmap[seqno]
                inputs = self.inputsFromOneFile(filename)
                rvalX[oib][ts] = inputs

            rvalY[oib] = numpy.asarray(self.seqmap[base_seqno])

        return rvalX, rvalY
