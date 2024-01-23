#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data_3_1 = np.load("mysterydata/mysterydata2.npy")
    for i in range(9):
        plt.imsave("images/vis_3_1_%d.png" % i,np.log(data_3_1[:,:,i] + 1.0))

    data_3_2 = np.load("mysterydata/mysterydata3.npy")
    minValue = np.nanmin(data_3_2)
    maxValue = np.nanmax(data_3_2)
    for i in range(9):
        plt.imsave("images/vis_3_2_%d.png" % i,data_3_2[:,:,i],vmin=minValue,vmax=maxValue)

    meanValue = np.mean(np.isfinite(data_3_2))
    print("Mean Value is: ",meanValue)
    print("Min Value is: ",minValue)
    print("Max Value is: ",maxValue)
    print("Bad value: ",data_3_2[10,10,0])


