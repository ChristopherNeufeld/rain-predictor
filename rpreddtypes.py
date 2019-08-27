#! /usr/bin/python3

import numpy as np
import gzip
import hashlib


# First, classes to manipulate the intermediate binary file.
#
# Format:
# RAIN PREDICTOR BIN FILE\n
# VERSION 3\n
# MAXVAL <num>
# TOTALRAIN <num>
# <Binary blob of byte values>



class RpBinFileReadError(Exception):
    def __init__(self, message):
        self.message = message


class RpBinObj:
    def __init__ (self):
        self.width = -1
        self.height = -1
        self.xoffset = -1
        self.yoffset = -1
        self.scaling = 1
        self.blen = -1
        self.buffer = None     # stored locally as a bytearray
        self.avgbuff = None
        self.magicnum = 0xa1b2
        self.version = 1

    def __read16bitInt(self, ba, index):
        return ba[index] * 256 + ba[index+1], index+2

    def __write16bitInt(self, value, ba, index):
        ba[index] = (value // 256) % 256
        ba[index + 1] = value % 256
        return index + 2

    def setvals(self, width, height, xoffset, yoffset,
                buffer):
        self.width = width
        self.height = height
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.scaling = 1
        self.buffer = buffer
        self.avgbuff = bytearray(len(buffer))
        for i in range(width * height):
            self.avgbuff[i] = 255

        self.blen = len(buffer)

    def getScaledVersion(self, edgeScale):
        rval = RpBinObj()
        newWidth = int((self.width - 1) // edgeScale + 1)
        newHeight = int((self.height - 1) // edgeScale + 1)
        rval.width = newWidth
        rval.height = newHeight
        rval.xoffset = self.xoffset
        rval.yoffset = self.yoffset
        rval.scaling = int(self.scaling * edgeScale)
        rval.blen = int(newWidth * newHeight)
        rval.buffer = bytearray(rval.blen)
        rval.avgbuff = bytearray(rval.blen)
        for i in range(newWidth):
            for j in range(newHeight):
                indexC = j * newWidth + i
                rval.buffer[indexC] = 0
                rval.avgbuff[indexC] = 255
                maxV = 0
                sumV = 0
                count = 0
                for deltaX in range(edgeScale):
                    for deltaY in range(edgeScale):
                        ii = i * edgeScale + deltaX
                        jj = j * edgeScale + deltaY
                        indexF = jj * self.width + ii
                        if ii < self.width and jj < self.height:
                            sumV += self.buffer[indexF]
                            count += 1
                            if self.buffer[indexF] > maxV:
                                maxV = self.buffer[indexF]

                if count == 0:
                    continue
                
                rval.buffer[indexC] = maxV
                avgV = sumV / count
                if maxV == 0:
                    maxV = 1
                    argV = 1
                rval.avgbuff[indexC] = int(avgV / maxV * 255)
        return rval


    def readFromByteArray(self, ba, index):
        mn, index = self.__read16bitInt(ba, index)
        if mn != self.magicnum:
            raise RpBinFileReadError("Incorrect magic number in RpBinObj")
        version, index = self.__read16bitInt(ba, index)
        if version != 1:
            raise RpBinFileReadError("Unrecognized version in RpBinObj")
        self.width, index = self.__read16bitInt(ba, index)
        self.height, index = self.__read16bitInt(ba, index)
        self.xoffset, index = self.__read16bitInt(ba, index)
        self.yoffset, index = self.__read16bitInt(ba, index)
        self.scaling, index = self.__read16bitInt(ba, index)

        self.blen = self.width * self.height
        self.buffer = ba[index:index + self.blen]
        index += self.blen
        self.avgbuff = ba[index:index + self.blen]
        return index + self.blen

    def writeToByteArray(self):
        rval = bytearray(7 * 2 + 2 * self.blen)
        index = 0
        index = self.__write16bitInt(self.magicnum, rval, index)
        index = self.__write16bitInt(self.version, rval, index)
        index = self.__write16bitInt(self.width, rval, index)
        index = self.__write16bitInt(self.height, rval, index)
        index = self.__write16bitInt(self.xoffset, rval, index)
        index = self.__write16bitInt(self.yoffset, rval, index)
        index = self.__write16bitInt(self.scaling, rval, index)
        rval[14:14 + self.blen] = self.buffer
        rval[14 + self.blen:] = self.avgbuff
        return rval

    def getNumpyArrayMax(self):
        return np.array(self.buffer).reshape([self.height, self.width])

    def getNumpyArrayAvg(self):
        maxes = self.getNumpyArrayMax()
        factors = np.array(self.avgbuff).reshape([self.height, self.width])
        return np.multiply(maxes, factors / 255)
    
    def getScaling(self):
        return self.scaling

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getXOffset(self):
        return self.xoffset

    def getYOffset(self):
        return self.yoffset

    def getVersion(self):
        return self.version




class RpBinCommon:
    HEADER_KEY = 'RAIN PREDICTOR BIN FILE'
    VERSION_KEY = 'VERSION'
    WIDTH_KEY = 'WIDTH'
    HEIGHT_KEY = 'HEIGHT'
    XOFFSET_KEY = 'XOFFSET'
    YOFFSET_KEY = 'YOFFSET'
    MAXVAL_KEY = 'MAXVAL'
    TOTALRAIN_KEY = 'TOTALRAIN'
    

class RpBinReader(RpBinCommon):
    """
    Reader for intermediate binary data type
    """
    def __init__ (self):
        self.version = 0
        self.maxval = 0     # Largest value in the byte payload(s)
        self.totalrain = -1
        self.numPayloads = -1
        self.elements = []

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
            if vnum != '3':
                raise RpBinFileReadError('File {0} is version {1}'
                                         'which is not supported'
                                         'by this code'.format(filename,
                                                               vnum))

            self.version = int(vnum)

            maxv, self.maxval = (istream.readline().rstrip()
                                 .decode('ascii').split(" "))
            self.maxval = int(self.maxval)
            if maxv != self.MAXVAL_KEY:
                raise RpBinFileReadError('File {0} is not a valid '
                                         'file'.format(filename))
            
            train, self.totalrain = (istream.readline().rstrip()
                                     .decode('ascii').split(" "))
            self.totalrain = int(self.totalrain)
            if train != self.TOTALRAIN_KEY:
                raise RpBinFileReadError('File {0} is not a valid '
                                         'file'.format(filename))

        if withData:
            btmp1 = istream.read()
            btmp2 = bytearray(gzip.decompress(btmp1))
            nbytes = len(btmp2)
            index = 0
            while index < nbytes:
                rpbo = RpBinObj()
                index = rpbo.readFromByteArray(btmp2, index)
                self.elements.append(rpbo)
            

    def getVersion(self):
        return self.version

    def getTotalRain(self):
        if self.version < 2:
            raise RpBinFileReadError('Total rain index is not supported '
                                     'by this file version.')
        return self.totalrain

    def getScaledObject(self, edgeScaling):
        for entry in self.elements:
            if entry.getScaling() == edgeScaling:
                return entry

        return None
            

class RpBinWriter(RpBinCommon):
    def __init__(self):
        pass

    def write(self, filename, width, height, xoffset, yoffset, maxval,
              totalRain, values, list_of_rescales = None):
        with open(filename, 'wb') as ofile:
            ofile.write('{0}\n'.format(self.HEADER_KEY)
                        .encode('ascii'))
            ofile.write('{0} {1}\n'.format(self.VERSION_KEY, 3)
                        .encode('ascii'))
            ofile.write('{0} {1}\n'.format(self.MAXVAL_KEY, maxval)
                        .encode('ascii'))
            ofile.write('{0} {1}\n'.format(self.TOTALRAIN_KEY, totalRain)
                        .encode('ascii'))

            rpbo = RpBinObj()
            rpbo.setvals(width, height, xoffset, yoffset, values)
            ebuffer = rpbo.writeToByteArray()
            for scale in list_of_rescales:
                nextobj = rpbo.getScaledVersion(scale)
                ebuffer += nextobj.writeToByteArray()
            
            ofile.write(gzip.compress(ebuffer))


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




def unittest(scratchfilename):
    testarray = np.array([ [ 0, 4, 3, 4, 9, 10 ],
                           [ 0, 2, 9, 7, 3, 1 ],
                           [ 2, 0, 3, 5, 3, 1 ],
                           [ 11, 8, 3, 1, 5, 6 ] ], 'int8')
    testW = 6
    testH = 4
    
    max2 = np.array([ [ 4, 9, 10 ],
                      [ 11, 5, 6 ] ], 'int8')
    avg2 = np.array([ [ 1.5, 23/4, 23/4 ],
                      [ 21/4, 3, 15/4 ] ])

    testW2 = 3
    testH2 = 2
    
    max3 = np.array([ [ 9, 10 ],
                      [ 11, 6 ] ], 'int8' )
    avg3 = np.array([ [ 23 / 9, 43 / 9 ],
                      [ 22 / 3, 4 ] ])

    testW3 = 2
    testH3 = 2

    writer = RpBinWriter()
    writer.write(scratchfilename, 6, 4, 0, 0, 11, 888,
                 bytearray(testarray.astype('int8')),
                 [ 2, 3 ])

    reader = RpBinReader()
    reader.read(scratchfilename)
    if reader.getTotalRain() != 888:
        print('Total rain match failure: {0} {1}'
              .format(reader.getTotalRain(), 888))
        return False


    obj1 = reader.getScaledObject(1)
    if not obj1:
        print('Failed to retrieve unscaled object.')
        return False

    if testW != obj1.getWidth():
        print('Unscaled width values match failure: {0} {1}'
              .format(testW, obj1.getWidth()))
        return False

    if testH != obj1.getHeight():
        print('Unscaled height values match failure: {0} {1}'
              .format(testH, obj1.getHeight()))
        return False

    if obj1.getScaling() != 1:
        print('Unscaled scaling factor reported incorrect: {0}'
              .format(obj1.getScaling()))
        return False

    if obj1.getXOffset() != 0:
        print('Unscaled XOffset reported incorrect: {0}'
              .format(obj1.getXOffset()))
        return False
        
    if obj1.getYOffset() != 0:
        print('Unscaled YOffset reported incorrect: {0}'
              .format(obj1.getYOffset()))
        return False


    check1 = obj1.getNumpyArrayMax()
    if not np.array_equal(testarray, check1):
        print('Unscaled max values match failure: {0} {1}'
              .format(check1, testarray))
        return False

    check1 = obj1.getNumpyArrayAvg()
    if not np.array_equal(testarray, check1):
        print('Unscaled avg values match failure: {0} {1}'
              .format(check1, testarray))
        return False

        


    obj2 = reader.getScaledObject(2)
    if not obj2:
        print('Failed to retrieve Scaled-2 object.')
        return False
    if testW2 != obj2.getWidth():
        print('Scaled-2 width values match failure: {0} {1}'
              .format(testW2, obj2.getWidth()))
        return False

    if testH2 != obj2.getHeight():
        print('Scaled-2 height values match failure: {0} {1}'
              .format(testH2, obj2.getHeight()))
        return False

    if obj2.getScaling() != 2:
        print('Scaled-2 scaling factor reported incorrect: {0}'
              .format(obj2.getScaling()))
        return False

    if obj2.getXOffset() != 0:
        print('Scaled-2 XOffset reported incorrect: {0}'
              .format(obj2.getXOffset()))
        return False
        
    if obj2.getYOffset() != 0:
        print('Scaled-2 YOffset reported incorrect: {0}'
              .format(obj2.getYOffset()))
        return False


    check2max = obj2.getNumpyArrayMax()
    if not np.array_equal(max2, check2max):
        print('Scaled-2 max values match failure: {0} {1}'
              .format(check2max, max2))
        return False

    # Remember, averages on scaled results aren't exact, but are
    # correct to within 1/255 of the max
    check2 = obj2.getNumpyArrayAvg()
    for i in range(obj2.getHeight()):
        for j in range(obj2.getWidth()):
            if abs(check2[i][j] - avg2[i][j]) > max2[i][j] / 255:
                print('Scaled-2 avg values match failure at {2},{3}: {0} {1}'
                      .format(check2, avg2, i, j))
                return False


    obj3 = reader.getScaledObject(3)
    if not obj3:
        print('Failed to retrieve Scaled-3 object.')
        return False
    if testW3 != obj3.getWidth():
        print('Scaled-3 width values match failure: {0} {1}'
              .format(testW3, obj3.getWidth()))
        return False

    if testH3 != obj3.getHeight():
        print('Scaled-3 height values match failure: {0} {1}'
              .format(testH3, obj3.getHeight()))
        return False

    if obj3.getScaling() != 3:
        print('Scaled-3 scaling factor reported incorrect: {0}'
              .format(obj3.getScaling()))
        return False

    if obj3.getXOffset() != 0:
        print('Scaled-3 XOffset reported incorrect: {0}'
              .format(obj3.getXOffset()))
        return False
        
    if obj3.getYOffset() != 0:
        print('Scaled-3 YOffset reported incorrect: {0}'
              .format(obj3.getYOffset()))
        return False


    check3max = obj3.getNumpyArrayMax()
    if not np.array_equal(max3, check3max):
        print('Scaled-3 max values match failure: {0} {1}'
              .format(check3max, max3))
        return False

    # Remember, averages on scaled results aren't exact, but are
    # correct to within 1/255 of the max
    check3 = obj3.getNumpyArrayAvg()
    for i in range(obj3.getHeight()):
        for j in range(obj3.getWidth()):
            if abs(check3[i][j] - avg3[i][j]) > max3[i][j] / 255:
                print('Scaled-3 avg values match failure: {0} {1}'
                      .format(check3[i][j], avg3[i][j]))
                return False


    return True
