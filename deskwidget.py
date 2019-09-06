#! /usr/bin/python3

import rpreddtypes
import argparse
import sys
import numpy
import random
# import keras
import tkinter


# Main code starts here

baseWidth = 200
baseHeight = 600
circlesize = 20


class BaseWidget(tkinter.Frame):
    def __init__(self):
        super().__init__()
        self.vals = [-1] * 10
        self.drawScreen()
        self.canvas = None

    def drawScreen(self):
        self.master.title("Rain Predictor")
        self.canvas = tkinter.Canvas(self)
        self.pack(fill=tkinter.BOTH, expand=1)
        self.canvas.create_text(baseWidth / 5,
                                baseHeight / 12,
                                anchor=tkinter.W,
                                text = "Any Rain")
        self.canvas.create_text(baseWidth * 7 / 10, baseHeight / 12,
                                text = "Heavy Rain")
        self.updateScreen()

    def updateScreen(self):
        print('self= {},  self.canvas= {}'.format(self, self.canvas))
        for time in range(5):
            for intensity in range(2):
                centreX = (1 + 2 * intensity) * baseWidth / 4
                centreY = (time + 1) * baseHeight / 6
                colour = None
                circleval = self.vals[2 * time + intensity]
                if circleval < 0:
                    colour = "black"
                elif circleval < 0.05:
                    colour = "green"
                elif circleval > 0.95:
                    colour = "red"
                else:
                    colour = "yellow"

                self.canvas.create_oval(centreX - circlesize / 2,
                                        centreY - circlesize / 2,
                                        centreX + circlesize / 2,
                                        centreY + circlesize / 2,
                                        outline = "#000000",
                                        fill=colour,
                                        width = 2)
        
        self.canvas.pack(fill=tkinter.BOTH, expand=1)
        print (self.canvas)


    def updateValues(self, vallist):
        self.vals = vallist.copy()
        print (self.canvas)
        self.updateScreen()


root = tkinter.Tk()

root.geometry('{0}x{1}'.format(baseWidth, baseHeight))
widget = BaseWidget()

def worker():
    print ('Hello')
    newlist = []
    for i in range(10):
        newlist.append(random.random())
    widget.updateValues(newlist)
    root.after(1000, worker)


worker()
root.mainloop()
