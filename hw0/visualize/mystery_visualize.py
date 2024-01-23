#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import pdb


def colormapArray(X, colors):
    """
    Basically plt.imsave but return a matrix instead

    Given:
        a HxW matrix X
        a Nx3 color map of colors in [0,1] [R,G,B]
    Outputs:
        a HxW uint8 image using the given colormap. See the Bewares
    """
    vmin = np.nanmin(X)
    vmax = np.nanmax(X)
    N = colors.shape[0]
    H = X.shape[0]
    W = X.shape[1]
    adjustedValues = (N-1) * (X-vmin) / (vmax-vmin)
    image = np.zeros((H,W,3))
    adjustedValues[np.isnan(adjustedValues)] = N-1

    for i in range(0,H):
        for j in range(0,W):
            image[i,j,:] = colors[int(adjustedValues[i,j])]

    return image #


if __name__ == "__main__":
    colors = np.load("mysterydata/colors.npy")
    data = np.load("mysterydata/mysterydata4.npy")

    for i in range(0,9):
        colorImage = colormapArray(data[:,:,i],colors)
        plt.imsave("images/vis_3_3_%d.png" % i,colorImage)


    #pdb.set_trace()
