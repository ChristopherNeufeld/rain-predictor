#! /usr/bin/python3

import matplotlib.pyplot as plt
import PIL

category = []
plt.ion()

images = []

with open('phantom-rain-candidates.txt', 'r') as ifile:
    filenames = ifile.readlines()

for fn in filenames:
    fn = fn[:-5]
    
    img = PIL.Image.open(fn)
    plt.imshow(img)
    plt.pause(0.05)
    response = input('PHANTOM?: ')
    plt.cla()
    if response == 'q':
        break
    category.append('{0}  {1}\n'.format(fn, response))


with open('screened-rain-candidates.txt', 'a+') as ofile:
    for result in category:
        ofile.write(result)
