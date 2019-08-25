#! /usr/bin/python3

# The data generator for the rain predictor
import keras
# from tensorflow import keras
import rpreddtypes
import math
import numpy


class RPDataGenerator(keras.utils.Sequence):
    'Produces data from the training set records we\'ve built'
    def __init__(self, sequence_file, path_file, veto_file,
                 centre, sensitive_region, heavyThreshold,
                 batch_size, shuffle=False):
        self.radius1 = 20
        self.radius2 = 60
        self.radius3 = 100
        self.radius4 = 170
        self.radius5 = 240
        self.tripradius = 25
        self.sequence_file = sequence_file
        self.path_file = path_file
        self.veto_file = veto_file
        self.centre = centre
        self.sensitive_region = sensitive_region
        self.heavy = heavyThreshold
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.hash = rpreddtypes.genhash(centre, sensitive_region,
                                        heavyThreshold)
        self.pathmap = {}
        self.seqmap = {}
        self.seqlist = []
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

        if self.shuffle:
            shuffleSequence()

    def shuffleSequence(self):
        random.shuffle(seqlist)

    def on_epoch_end(self):
        self.shuffleSequence()

    def __len__(self):
        return 1 + ( len(self.seqlist) // self.batch_size )

    def buildModules(self):
        # For each module, we store a list of pixels.  We put the
        # centre in its module.  Then, starting at the centre, we walk
        # up to the outer radius, putting all points in a module.  We
        # then go one pixel to the right, and, starting one pixel
        # above the centre, do it again.  Continue until we've run out
        # of space to the right.  Then we rotate the pixels 90 degrees
        # into their new modules.  This guarantees no missed pixels.

        self.modules = [[] for _ in range(34)]

        # Modules are numbered clockwise from the outermost, to the
        # right of the centreline, 0.  After 7, we go inward one ring
        # and repeat, starting to the right of the centreline.  This
        # covers modules 0-31.  Module 32 is the bullseye, and Module
        # 33 is the tripwire.

        self.modules[32].append(self.centre)

        for i in range(self.centre[0], 2 * self.centre[0]):
            for j in range(self.centre[1], 0, -1):
                deltaI = i - self.centre[0]
                deltaJ = self.centre[1] - j
                distance = math.sqrt(deltaI ** 2 + deltaJ ** 2)
                if distance <= self.radius1:
                    self.modules[32].append([i, j])
                elif distance <= self.radius2:
                    if deltaJ < deltaI:
                        self.modules[24].append([i, j])
                    else:
                        self.modules[25].append([i, j])
                elif distance <= self.radius3:
                    if deltaJ < deltaI:
                        self.modules[16].append([i, j])
                    else:
                        self.modules[17].append([i, j])
                elif distance <= self.radius4:
                    if deltaJ < deltaI:
                        self.modules[8].append([i, j])
                    else:
                        self.modules[9].append([i, j])
                elif distance <= self.radius5:
                    if deltaJ < deltaI:
                        self.modules[0].append([i, j])
                    else:
                        self.modules[1].append([i, j])
                else:
                    break


        nBullseye = len(self.modules[32])
                
        # Now we've loaded one quadrant.  Copy with rotation to the
        # other three quadrants.
        for i in range(6):
            for ring in range(4):
                self.modules[ring * 8 + i + 2] = list(map(lambda p: [ p[1], -p[0]], self.modules[ring * 8 + i]))

            # Do the bullseye
            for p in range(nBullseye):
                pc = self.modules[32][i * nBullseye + p]
                self.modules[32].append([pc[1], -pc[0]])

        
        # Finally, the tripwire
        for radius in range(-self.tripradius, self.tripradius + 1):
            i1 = self.centre[0] + radius
            j1 = self.centre[1] + int(math.sqrt(self.tripradius ** 2 - radius ** 2))
            j2 = 2 * self.centre[1] - j1
            if not [i1, j1] in self.modules[33]:
                self.modules[33].append([i1, j1])
            if not [i1, j2] in self.modules[33]:
                self.modules[33].append([i1, j2])

            j1 = self.centre[1] + radius
            i1 = self.centre[0] + int(math.sqrt(self.tripradius ** 2 - radius ** 2))
            i2 = 2 * self.centre[0] - i1
            if not [i1, j1] in self.modules[33]:
                self.modules[33].append([i1, j1])
            if not [i2, j1] in self.modules[33]:
                self.modules[33].append([i2, j1])


    def getModuleSizes(self):
        return list(map(len, self.modules))

    def normalize(self, val):
        if (val < self.heavy):
            return val / 20.0
        else:
            return (5 + val) / 20.0


    def inputsFromOneFile(self, filename):
        reader = rpreddtypes.RpBinReader()
        reader.read(filename)
        sourcepixels = reader.getNumpyArray()
        rval = []
        for module in range(34):
            nPixels = len(self.modules[module])
            rval.append(numpy.empty(nPixels))
            for pixelIndex in range(nPixels):
                pixel = self.modules[module][pixelIndex]
                rval[module][pixelIndex] = self.normalize(sourcepixels[pixel[0]][pixel[1]])
                
        return rval

        
        
    def __getitem__(self, index):
        'Return one batch'
        # For each module, have to return a numpy array
        # indexed by [offsetInBatch][timestep][pixelIndex]
        # Hand these back in a list, and the y-vals in another list

        rvalX = []
        rvalY = numpy.empty([self.batch_size, 10])
        for i in range(34):
            rvalX.append(numpy.empty([self.batch_size, 6,
                                      len(self.modules[i])]))

        for oib in range(self.batch_size):
            base_seqno = self.seqlist[index * self.batch_size + oib]
            for ts in range(6):
                seqno = base_seqno + ts
                filename = self.pathmap[seqno]
                inputs = self.inputsFromOneFile(filename)
                for m in range(34):
                    rvalX[m][oib][ts] = inputs[m]

            rvalY[oib] = numpy.asarray(self.seqmap[base_seqno])

        return rvalX, rvalY
