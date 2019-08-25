#! /usr/bin/python3

import numpy as np
import gzip
import hashlib


# First, classes to manipulate the intermediate binary file.
#
# Format:
# RAIN PREDICTOR BIN FILE\n
# VERSION 1\n
# WIDTH NNN\n
# HEIGHT NNN\n
# XOFFSET NNN\n
# YOFFSET NNN\n
# <Binary blob of byte values>



class RpBinFileReadError(Exception):
    def __init__(self, message):
        self.message = message

class RpBinCommon:
    HEADER_KEY = 'RAIN PREDICTOR BIN FILE'
    VERSION_KEY = 'VERSION'
    WIDTH_KEY = 'WIDTH'
    HEIGHT_KEY = 'HEIGHT'
    XOFFSET_KEY = 'XOFFSET'
    YOFFSET_KEY = 'YOFFSET'
    TOTALRAIN_KEY = 'TOTALRAIN'
    

class RpBinReader(RpBinCommon):
    """
    Reader for intermediate binary data type
    """
    def __init__ (self,):
        self.version = 0
        self.buffer = b''
        self.blen = 0
        self.width = 0
        self.height = 0
        self.xoffset = 0
        self.yoffset = 0
        self.totalrain = -1

    def read(self, filename):
        self.readHeader(filename, True)
        

    def readHeader(self, filename, withData = False):
        with open(filename, 'rb') as istream:
            try:
                istream = open(filename, 'rb')
            except OSError as ex:
                raise RpBinFileReadError(ex.strerror)

            header = istream.readline().rstrip().decode('ascii')
            if header != self.HEADER_KEY:
                raise RpBinFileReadError('File {0} is not a valid '
                                         'file'.format(filename))
            
            vstr, vnum = (istream.readline().rstrip()
                          .decode('ascii').split(" "))
            if vstr != self.VERSION_KEY:
                raise RpBinFileReadError('File {0} is not a valid '
                                         'file'.format(filename))
            if vnum != '1' and vnum != '2':
                raise RpBinFileReadError('File {0} is version {1}'
                                         'which is not supported'
                                         'by this code'.format(filename,
                                                               vnum))

            self.version = int(vnum)

            width, self.width = (istream.readline().rstrip()
                                 .decode('ascii').split(" "))
            self.width = int(self.width)
            if width != self.WIDTH_KEY or self.width < 0:
                raise RpBinFileReadError('File {0} is not a valid '
                                         'file'.format(filename))

            height, self.height = (istream.readline().rstrip()
                                   .decode('ascii').split(" "))
            self.height = int(self.height)
            if width != self.WIDTH_KEY or self.height < 0:
                raise RpBinFileReadError('File {0} is not a valid '
                                         'file'.format(filename))

            xoffset, self.xoffset = (istream.readline().rstrip()
                                     .decode('ascii').split(" "))
            self.xoffset = int(self.xoffset)
            if xoffset != self.XOFFSET_KEY or self.xoffset < 0:
                raise RpBinFileReadError('File {0} is not a valid '
                                         'file'.format(filename))

            yoffset, self.yoffset = (istream.readline().rstrip()
                                     .decode('ascii').split(" "))
            self.yoffset = int(self.yoffset)
            if yoffset != self.YOFFSET_KEY or self.yoffset < 0:
                raise RpBinFileReadError('File {0} is not a valid '
                                         'file'.format(filename))

            if vnum == '2':
                train, self.totalrain = (istream.readline().rstrip()
                                         .decode('ascii').split(" "))
                self.totalrain = int(self.totalrain)
                if train != self.TOTALRAIN_KEY:
                    raise RpBinFileReadError('File {0} is not a valid '
                                             'file'.format(filename))

        if withData:
            btmp = istream.read()
            self.buffer = gzip.decompress(btmp)
            self.blen = self.width * self.height
            if self.blen != len(self.buffer):
                raise RpBinFileReadError('File {0} is '
                                         'corrupted'.format(filename))
            

    def getVersion(self):
        return self.version

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getXOffset(self):
        return self.xoffset

    def getYOffset(self):
        return self.yoffset

    def getTotalRain(self):
        if self.version < 2:
            raise RpBinFileReadError('Total rain index is not supported '
                                     'by this file version.')
        return self.totalrain
            

    def get1Dbuffer(self):
        if not self.buffer:
            raise RpBinFileReadError('Data buffer not available, maybe you '
                                     'only read the header.')
        return self.buffer

    def getNumpyArray(self):
        if not self.buffer:
            raise RpBinFileReadError('Data buffer not available, maybe you '
                                     'only read the header.')

        array = np.arange(self.width * self.height, dtype=int)
        for i in range(self.width * self.height):
            array[i] = self.buffer[i]

        return array.reshape(self.width, self.height)

    

class RpBinWriter(RpBinCommon):
    def __init__(self):
        pass

    def write(self, filename, width, height, xoffset, yoffset,
              totalRain, values):
        with open(filename, 'wb') as ofile:
            ofile.write('{0}\n'.format(self.HEADER_KEY)
                        .encode('ascii'))
            ofile.write('{0} {1}\n'.format(self.VERSION_KEY, 2)
                        .encode('ascii'))
            ofile.write('{0} {1}\n'.format(self.WIDTH_KEY, width)
                        .encode('ascii'))
            ofile.write('{0} {1}\n'.format(self.HEIGHT_KEY, height)
                        .encode('ascii'))
            ofile.write('{0} {1}\n'.format(self.XOFFSET_KEY, xoffset)
                        .encode('ascii'))
            ofile.write('{0} {1}\n'.format(self.YOFFSET_KEY, yoffset)
                        .encode('ascii'))
            ofile.write('{0} {1}\n'.format(self.TOTALRAIN_KEY, totalRain)
                        .encode('ascii'))
            ofile.write(gzip.compress(bytearray(values)))


def genhash(centre, senseList, heavyThreshold, seed = 0xabcddcba):
    """
    Returns a string, a hash derived from MD5 of the inputs
    """

    hasher = hashlib.md5()
    hasher.update('{0:0>8x}'.format(seed).encode('ascii'))
    hasher.update('{0:0>8x}{1:0>8x}'
                  .format(centre[0], centre[1])
                  .encode('ascii'))
    for tuple in senseList:
        hasher.update('{0:0>8x}{1:0>8x}'
                      .format(tuple[0], tuple[1])
                      .encode('ascii'))

    hasher.update('{0:0>8x}'.format(heavyThreshold).encode('ascii'))
        
    return hasher.digest().hex()[0:8]


