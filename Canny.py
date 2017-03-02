#!/usr/bin/python

# Brandon Sangston bss3cv
# CS 4501 Computer Vision
# 2/16/2017
# Canny.py - implements Canny Edge Detector, taking in a image file path as a command line argument, output saved
# to canny_output.png

import sys
import numpy as np
import pylab
import scipy.ndimage.filters as ndi
import skimage.io
from numba import jit

# Convert to greyscale 2D array

def rgb_avg(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

@jit
def greyscale(image):
    gray = np.zeros((image.shape[0], image.shape[1]))
    for row in range(len(image)):
        for col in range(len(image[row])):
            gray[row, col] = rgb_avg(image[row,col])
    return gray


# Blur image with first derivative of gaussian filter
def gaussian1D(n, sigma=3.0):
    result = np.zeros(n)
    mid = n / 2
    result = [(1 / (np.sqrt(2 * np.pi * sigma**2)) * (np.e**(-((i**2) / (2 * sigma**2))))) for i in
              range(-mid, mid + 1)]

    return result

@jit
def gradient_magnitude(image, dx, dy):
    orientations = np.zeros(image.shape)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            image[y][x] = np.sqrt(dx[y][x]**2 + dy[y][x]**2)
            orientations[y][x] = np.arctan2(dy[y][x],dx[y][x])
    return image, orientations


pi = np.pi

@jit
def nonmax_suppression(image, orientations):

    for y in range(1, image.shape[0]-1):
        for x in range(1, image.shape[1]-1):

            if (0 <= abs(orientations[y][x]) <= pi/8 or 15*pi/8 <= abs(orientations[y][x]) <= 2*pi or
                    7*pi/8 <= abs(orientations[y][x]) <= 9*pi/8):
                if (image[y][x] < image[y+1][x]) or \
                   (image[y][x] < image[y-1][x]):
                    image[y][x] = 0

            if 3*pi/8 <= abs(orientations[y][x]) <= 5*pi/8 or 11*pi/8 <= abs(orientations[y][x]) <= 13*pi/8:
                if (image[y][x] < image[y][x + 1]) or \
                   (image[y][x] < image[y][x - 1]):
                    image[y][x] = 0

            if (pi/8 < abs(orientations[y][x]) < 3*pi/8) or (9*pi/8 < abs(orientations[y][x]) < 11*pi/8):
                if (image[y][x] < image[y + 1][x + 1]) or \
                   (image[y][x] < image[y - 1][x - 1]):
                    image[y][x] = 0

            if 5*pi/8 < abs(orientations[y][x]) < 7*pi/8 or 13*pi/8 < abs(orientations[y][x]) < 15*pi/8:
                if (image[y][x] < image[y + 1][x - 1]) or \
                   (image[y][x] < image[y - 1][x + 1]):
                    image[y][x] = 0

    return image

@jit
def hyst_link(image, x, y, visited, T_low):

    if 1 < x < image.shape[0]-1 and 1 < y < image.shape[1]-1 and \
            1 < x < visited.shape[0] - 1 and 1 < y < visited.shape[1] - 1:

        if image[x + 1][y + 1] >= T_low and visited[x+1][y+1] != 1:
            image[x + 1][y + 1] = 1
            visited[x + 1][y + 1] = 1
            hyst_link(image, x+1,y+1,visited, T_low)

        if image[x + 1][y] >= T_low and visited[x+1][y] != 1:
            image[x + 1][y] = 1
            visited[x + 1][y] = 1
            hyst_link(image, x + 1, y, visited, T_low)

        if image[x + 1][y - 1] >= T_low and visited[x+1][y-1] != 1:
            image[x + 1][y - 1] = 1
            visited[x + 1][y - 1] = 1
            hyst_link(image, x + 1, y - 1, visited, T_low)

        if image[x - 1][y + 1] >= T_low and visited[x-1][y+1] != 1:
            image[x - 1][y + 1] = 1
            visited[x - 1][y + 1] = 1
            hyst_link(image, x - 1, y + 1, visited, T_low)

        if image[x - 1][y] >= T_low and visited[x-1][y] != 1:
            image[x - 1][y] = 1
            visited[x - 1][y] = 1
            hyst_link(image, x - 1, y, visited, T_low)

        if image[x - 1][y - 1] >= T_low and visited[x-1][y-1] != 1:
            image[x - 1][y - 1] = 1
            visited[x - 1][y - 1] = 1
            hyst_link(image, x - 1, y - 1, visited, T_low)

        if image[x][y + 1] >= T_low and visited[x][y+1] != 1:
            image[x][y + 1] = 1
            visited[x][y + 1] = 1
            hyst_link(image, x, y + 1, visited, T_low)

        if image[x][y - 1] >= T_low and visited[x][y+1] != 1:
            image[x][y - 1] = 1
            visited[x][y - 1] = 1
            hyst_link(image, x, y - 1, visited, T_low)

@jit
def hyst_threshold(image, T_low=0.1, T_high=0.3):
    visited = np.zeros(image.shape)

    for x in range(1, image.shape[0]-1):
        for y in range(1, image.shape[1]-1):
            if image[x][y] > T_high and visited[x][y] != 1:
                image[x][y] = 1
                visited[x][y] = 1
                hyst_link(image, x, y, visited, T_low)

            if image[x][y] < T_low and visited[x][y] != 1:
                image[x][y] = 0
                visited[x][y] = 1

            if T_low >= image[x][y] >= T_high and visited[x][y] != 1:
                image[x][y] = 0.5

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y] != 1:
                image[x][y] = 0

    return image


def main(args):
    if len(sys.argv) == 1:
        print 'Error: No valid image file path entered as command line argument.' \
              ' Please enter a valid image path to continue.'
        sys.exit()

    print "Finding edges..."
    I = skimage.img_as_float(skimage.io.imread(args[1]))

    I_grey = greyscale(I)

    sigma = 1
    kernel_size = 9
    H = np.array(gaussian1D(kernel_size, sigma))
    V = np.copy(H)

    dv = np.array([-1,0,1])

    H_dx = ndi.convolve1d(H, dv)
    V_dy = ndi.convolve1d(V, dv)

    G_x = ndi.convolve1d(I_grey, H_dx, 0)
    G_y = ndi.convolve1d(I_grey, V_dy, 1)

    G = np.zeros(I_grey.shape)
    F, orientations = gradient_magnitude(G, G_x, G_y)
    F_thinned = np.copy(F)

    F_thinned = nonmax_suppression(F_thinned, orientations)

    thresh = 0.33

    if len(args) > 2:
        thresh = float(args[2])

    T_high = thresh * np.max(F_thinned)
    T_low = 0.5 * T_high

    F_t = np.copy(F_thinned)

    F_final = hyst_threshold(F_t, T_low, T_high)

    pylab.figure(1)
    pylab.imshow(F_final, cmap="gray")

    skimage.io.imsave('./canny_output.png', F_final.astype('float32'))

    pylab.show()

if __name__ == '__main__':
    main(sys.argv)



