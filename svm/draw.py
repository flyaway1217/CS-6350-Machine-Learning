# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.1
#
# Date: 2016-09-24 09:09:15
# Last modified: 2016-11-14 21:20:16

"""
According to the data, draw the 3D picture.
"""


import random

# import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mpl_toolkits.mplot3d import Axes3D


def read(path):
    values = []
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.split()
            if len(s) != 0:
                values.append([float(v) for v in s])

    x = [v[0] for v in values]
    y = [v[1] for v in values]
    z = [v[4] for v in values]

    return x, y, z


def draw3D(x, y, z):
    mpl.rcParams['font.size'] = 10
    fig = plt.figure()
    ax = fig.add_subplot(111,  projection='3d')
    color = [plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
             for i in range(len(x))]
    ax.bar(y,  z,  zs=x,  zdir='x',  color=color,
           alpha=0.8, bottom=0, width=y[1]-y[0])

    ax.set_xlabel('Rate')
    ax.set_ylabel('C')
    ax.set_zlabel('Recall')
    plt.show()


def draw2D(x, z):
    # plt.plot(x, z)
    color = ['r', 'b'] * int(len(x)/2)
    z = [(v-0.7)*100 for v in z]
    plt.bar(x, z, alpha=0.4, width=x[1]-x[0], color=color, bottom=70)
    plt.xlabel('Rate')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == '__main__':
    x, y, z = read('./simples.txt')
    # print(y)
    # draw(x[0:6], y[0:6], z[0:6])
    draw3D(x, y, z)
    # draw2D(y, z)
    # draw2D(x, z)
