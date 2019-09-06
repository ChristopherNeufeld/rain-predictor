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


class BaseWidget():
    def __init__(self, root):
        self.root = root
        root.title("Rain Predictor")
        self.canvas = tkinter.Canvas(root,
                                     bg = "Black",
                                     width = baseWidth,
                                     height = baseHeight)
        self.canvas.pack()
        self.vals = [-1] * 10
        self.drawScreen()

    def drawScreen(self):
        self.canvas.create_text(baseWidth / 6,
                                baseHeight / 12,
                                anchor=tkinter.W,
                                fill = "White",
                                text = "Any Rain")
        self.canvas.create_text(baseWidth * 7 / 10, baseHeight / 12,
                                fill = "White",
                                text = "Heavy Rain")
        self.canvas.pack()
        self.updateScreen()

    def updateScreen(self):
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
                                        outline = "White",
                                        fill=colour,
                                        width = 2)
        
        self.canvas.pack(fill=tkinter.BOTH, expand=1)


    def updateValues(self, vallist):
        self.vals = vallist.copy()
        self.updateScreen()


def worker(screen):
    print ('Hello')
    newlist = []
    for i in range(10):
        newlist.append(random.random())
    screen.updateValues(newlist)
    root.after(1000, worker, screen)


root = tkinter.Tk()

root.geometry('{0}x{1}'.format(baseWidth, baseHeight))
widget = BaseWidget(root)

root.after(1000, worker, widget)
root.mainloop()
