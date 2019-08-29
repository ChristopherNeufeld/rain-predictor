#! /usr/bin/python3

# The data generator for the rain predictor
import keras
# from tensorflow import keras
import rpreddtypes
import math
import numpy
import random


class RPDataGenerator2(keras.utils.Sequence):
    'Produces data from the training set records we\'ve built'
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
        self.num_intensities = None
        self.intensity_gap = 5    # to separate light from heavy rain
                                  # in the inputs
        self.batch_size = batch_size
        self.hash = rpreddtypes.genhash(centre, sensitive_region,
                                        heavyThreshold)
        self.pathmap = {}
        self.seqmap = {}
        self.seqlist = []
        self.numModules = 0
        self.modules = []

        self.buildModules()
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

        self.shuffleSequence()

    def shuffleSequence(self):
        random.shuffle(self.seqlist)

    def on_epoch_end(self):
        self.shuffleSequence()

    def __len__(self):
        return len(self.seqlist) // self.batch_size

    def getInputSize(self):
        return 2 * len(self.modules)

    def buildModules(self):
        # For each module, we store a list of pixels.  We walk the
        # pixel space and assign each one to a single module.  This
        # does not, in general, guarantee that the modules have
        # exactly the same number of pixels, but that's not an
        # important consideration here.

        self.numModules = self.ringcount * self.radialcuts
        self.modules = numpy.full((2 * self.centre[0],
                                   2 * self.centre[0]), -1)

        # Modules are numbered clockwise (because this is a
        # left-handded coordinate system) from the X axis, closest
        # ring first.

        radius = self.centre[0]
        r2 = radius ** 2
        for pixelI in range(2 * radius):
            for pixelJ in range(2 * radius):
                delta = [ pixelI - self.centre[0], pixelJ - self.centre[1] ]
                d2 = delta[0] ** 2 + delta[1] ** 2
                if d2 > r2:
                    continue

                ringnum = int(math.sqrt(d2) / radius * self.ringcount)
                if ringnum >= self.ringcount:
                    continue  # shouldn't happen for reasonable width floats

                angle = math.atan2(delta[1], delta[0])
                if angle < 0:
                    angle += 2 * math.pi

                secnum = int(angle / (2 * math.pi) * self.radialcuts)

                # More "shouldn't happen" floating point defense
                if secnum < 0:
                    secnum = 0
                if secnum >= self.radialcuts:
                    secnum = self.radialcuts

                modnum = int(ringnum * self.radialcuts + secnum)
                self.modules[pixelJ, pixelI] = modnum


    def inputsFromOneFile(self, filename):
        reader = rpreddtypes.RpBinReader()
        reader.read(filename)
        if not self.num_intensities:
            self.num_intensities = reader.getMaxRainval()

        rpbo = reader.getPreparedDataObject()
        return numpy.asarray(rpbo.getPreparedData()) / 255

        
    def __getitem__(self, index):
        'Return one batch'
        # Return xvals in a numpy array
        # indexed by [offsetInBatch][timestep][valindex]
        # Hand the y-vals back in another array

        print ('Loading one batch\n')
        rvalX = numpy.empty([self.batch_size, 6, 2 * self.numModules])
        rvalY = numpy.empty([self.batch_size, 10])

        for oib in range(self.batch_size):
            print ('.', end='', flush=True)
            base_seqno = self.seqlist[index * self.batch_size + oib]
            for ts in range(6):
                seqno = base_seqno + ts
                filename = self.pathmap[seqno]
                inputs = self.inputsFromOneFile(filename)
                rvalX[oib][ts] = inputs

            rvalY[oib] = numpy.asarray(self.seqmap[base_seqno])

        print ('Batch loaded\n')
        return rvalX, rvalY
