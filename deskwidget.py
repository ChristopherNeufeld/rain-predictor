#! /usr/bin/python3

import rpreddtypes
import argparse
import sys
import numpy as np
import random
import keras
import tkinter
import os
import configparser
import datetime
import threading


# Main code starts here

baseWidth = 200
baseHeight = 600
circlesize = 20
haloColour = "White"
clearColour = "Green"
uncertainColour = "Yellow"
rainColour = "Red"
canvasColour = "Black"
refreshTime = 1000

savedNetwork = None
gifFileFormatString = None
binFileFormatString = None


class BaseWidget():
    def __init__(self, root, rwidth, rheight, csize, hcolour,
                 noRainColour, maybeRainColour, yesRainColour, canvasBg,
                 loopTime, network, gifFormat, binFormat, lockfile):
        self.root = root
        self.width = rwidth
        self.height = rheight
        self.csize = csize
        self.hcolour = hcolour
        self.noRainColour = noRainColour
        self.maybeRainColour = maybeRainColour
        self.yesRainColour = yesRainColour
        self.canvasBg = canvasBg
        self.loopTime = loopTime
        self.network = network
        self.gifFormat = gifFormat
        self.binFormat = binFormat
        self.lockfile = lockfile
        
        self.lastUpdate = tkinter.StringVar()
        self.lastUpdate.set('NO LAST UPDATE')
        self.gifFileNames = [ None ] * 6
        self.binFileNames = [ None ] * 6
        self.needRefresh = 1
        self.gwin = None
        self.giflabel = None

        self.mutex = threading.Lock()

        root.title("Rain Predictor")
        self.gifbutton = tkinter.Button(root, bg = "White", text="Show GIFs",
                                        command=self.makeGifWindow)
        self.gifbutton.pack()
        self.updateLabel = tkinter.Label(root, fg = "Black", 
                                         textvariable = self.lastUpdate)
        self.updateLabel.pack()
        self.canvas = tkinter.Canvas(root,
                                     bg = canvasBg,
                                     width = self.width,
                                     height = self.height - 40)
        self.canvas.pack()
        self.vals = [-1] * 10
        self.gifImages = [ None ] * 6
        self.gifnum = 0
        self.validDate = ''  # last time we had valid data

        self.drawScreen()

    def makeGifWindow(self):
        self.mutex.acquire()
        self.gwin = tkinter.Toplevel()
        self.gwin.wm_title("Radar Images")
        for i in range(6):
            self.gifImages[i] = tkinter.PhotoImage(file = self.gifFileNames[i],
                                                   format = "gif")
        self.giflabel = tkinter.Label(self.gwin,
                                      height = self.gifImages[0].height(),
                                      width = self.gifImages[0].width(),
                                      image = self.gifImages[0])
        self.giflabel.pack()
        self.gwin.protocol("WM_DELETE_WINDOW", self.closeGifWindow)
        self.mutex.release()
        

    def closeGifWindow(self):
        self.mutex.acquire()
        self.gwin.destroy()
        self.giflabel = None
        self.gwin = None
        self.gifImages = [ None ] * 6
        self.gifnum = 0
        self.mutex.release()


    def rotateGifs(self):
        self.mutex.acquire()
        if not self.gwin:
            self.mutex.release()
            return
        self.gifnum = (self.gifnum + 1) % 6
        self.giflabel.configure(image = self.gifImages[self.gifnum])
        self.mutex.release()
        
        
    def drawScreen(self):
        self.canvas.create_text(self.width * 0.30,
                                self.height * 0.08,
                                fill = "White",
                                text = "Any Rain")
        self.canvas.create_text(self.width * 0.7, self.height * 0.08,
                                fill = "White",
                                text = "Heavy Rain")

        for i in range(5):
            self.canvas.create_text(0.1 * self.width,
                                    (i + 1) / 6 * self.height,
                                    fill = "White",
                                    text = '{0}-{1}'.format(i, i+1))
        
        self.canvas.pack()
        self.updateScreen()

    def updateScreen(self):
        if self.needRefresh == 0:
            return
        self.needRefresh = 0
        for time in range(5):
            for intensity in range(2):
                centreX = (0.35 + 0.45 * intensity) * self.width
                centreY = (time + 1) / 6 * self.height
                colour = None
                circleval = self.vals[2 * time + intensity]
                if circleval < 0:
                    colour = "black"
                elif circleval < 0.05:
                    colour = self.noRainColour
                elif circleval > 0.95:
                    colour = self.yesRainColour
                else:
                    colour = self.maybeRainColour

                self.canvas.create_oval(centreX - self.csize / 2,
                                        centreY - self.csize / 2,
                                        centreX + self.csize / 2,
                                        centreY + self.csize / 2,
                                        outline = self.hcolour,
                                        fill=colour,
                                        width = 2)
        
        self.canvas.pack(fill=tkinter.BOTH, expand=1)


    def loadNewBinFiles(self):
        now = datetime.datetime.utcnow()
        nowYear = now.year
        nowMonth = now.month
        nowDay = now.day
        nowHour = now.hour

        # round to previous 10-minute interval
        nowMinute = int( now.minute / 10 ) * 10

        now = now.replace(minute = nowMinute)

        if self.lockfile and os.path.exists(self.lockfile):
            return

        for retry in range(3):
            gooddat = True
            for inum in range(6):
                delta = datetime.timedelta(0, 600 * (retry + 5 - inum))
                dtime = now - delta
                dYear = dtime.year
                dMonth = dtime.month
                dDay = dtime.day
                dHour = dtime.hour
                dMin = dtime.minute
                self.binFileNames[inum] = ( self.binFormat.format(YEAR = dYear, MONTH = dMonth, DAY = dDay, HOUR = dHour, MIN = dMin) )
                self.gifFileNames[inum] = ( self.gifFormat.format(YEAR = dYear, MONTH = dMonth, DAY = dDay, HOUR = dHour, MIN = dMin) )
                if not os.path.exists(self.binFileNames[inum]):
                    if retry == 2:
                        if now - self.validDate > datetime.timedelta(0, 1800):
                            self.updateLabel.configure(bg = "Yellow")
                        return

                    gooddat = False

            if gooddat:
                thisstring = ('{YEAR:04d}_{MONTH:02d}_{DAY:02d}_{HOUR:02d}_{MINUTE:02d}'
                              .format(YEAR=dYear,
                                      MONTH=dMonth,
                                      DAY=dDay,
                                      HOUR=dHour,
                                      MINUTE=dMin))

                break


        if thisstring == self.lastUpdate:
            return

        reader = rpreddtypes.RpBinReader()
        reader.read(self.binFileNames[0])
        rpbo = reader.getPreparedDataObject()
        datasize = rpbo.getDataLength()

        xvals = np.empty([1, 6, datasize])

        for timestep in range(6):
            reader = rpreddtypes.RpBinReader()
            reader.read(self.binFileNames[timestep])
            rpbo = reader.getPreparedDataObject()
            xvals[0][timestep] = np.asarray(rpbo.getPreparedData()) / 255
            
        self.vals = self.network.predict(xvals)[0]
        self.lastUpdate.set(thisstring)
        self.validDate = now
        self.updateLabel.configure(bg = "White")
        self.needRefresh = 1
        

    def updateValues(self):
        try:
            self.loadNewBinFiles()
            self.updateScreen()
        except Exception as ex:
            return


def worker(screen):
    screen.updateValues()
    screen.rotateGifs()
    root.after(screen.loopTime, worker, screen)



configPath = os.environ.get('RAIN_PREDICTOR_CONF')
if not configPath:
    configPath = ".rpwidget.conf"

config = configparser.ConfigParser()
config.read_file(open(configPath))
if not config.sections():
    print ('Unable to load configuration.')
    sys.exit(1)

baseWidth = int(config.get('Graphics', 'Width', fallback=baseWidth))
baseHeight = int(config.get('Graphics', 'Height', fallback=baseHeight))
circlesize = int(config.get('Graphics', 'LightSize', fallback=circlesize))
haloColour = config.get('Graphics', 'LightHaloColour',
                        fallback=haloColour)
clearColour = config.get('Graphics', 'LightNoRainColour',
                         fallback=clearColour)
uncertainColour = config.get('Graphics', 'LightMaybeRainColour',
                             fallback=uncertainColour)
rainColour = config.get('Graphics', 'LightYesRainColour',
                        fallback=rainColour)
refreshTime = int(config.get('Graphics', 'GifSpeedMs',
                             fallback=refreshTime))

savedNetwork = config.get('Network', 'SavedNetwork')
gifFileFormatString = config.get('Data', 'GifFiles')
binFileFormatString = config.get('Data', 'BinFiles')
lockfile = config.get('Data', 'LockFile')

if not savedNetwork:
    print('Need a saved network for producing predictions')
    sys.exit(1)

if not gifFileFormatString:
    print ('Need a gif file format string for graphical display')
    sys.exit(1)

if not binFileFormatString:
    print ('Need a bin file format string to load data for the predictions')
    sys.exit(1)
        
    

# now = datetime.datetime.utcnow()
# nowYear = now.year
# nowMonth = now.month
# nowDay = now.day
# nowHour = now.hour
# nowMinute = int( now.minute / 10 ) * 10   # round to previous 10-minute interval


network = keras.models.load_model(savedNetwork)
    
root = tkinter.Tk()

root.geometry('{0}x{1}'.format(baseWidth, baseHeight))
widget = BaseWidget(root, baseWidth, baseHeight, 
                    circlesize, haloColour, clearColour, uncertainColour,
                    rainColour, canvasColour, refreshTime,
                    network, gifFileFormatString, binFileFormatString,
                    lockfile)

root.after(refreshTime, worker, widget)
root.mainloop()
