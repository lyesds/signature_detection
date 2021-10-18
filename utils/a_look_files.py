import random

import cv2 as cv
import glob
import random
import numpy as np
#from matplotlib import pyplot as plt

train = glob.glob('data/train/*.tif')
test = glob.glob('data/test/*.tif')

k=3
#print(random.choices(train, k=k), random.choices(test, k=k))

def rescale_and_show_random():
    for i in random.choices(train, k=k):
        img= cv.imread(i, -1)
        print(i)
        print('Original Dimensions : ', img.shape)

        scale_percent = 20  # percent of original size
        if img.shape[0]<2000:
            scale_percent = 40
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        dim = (480, 630)

        # resize image
        resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

        print('Resized Dimensions : ', resized.shape)

        adjusted1 = cv.convertScaleAbs(resized, alpha=2, beta=0)
        #cv.imshow('original', img)
        cv.imshow('resized', resized)
        #cv.imshow('adjusted', adjusted1)
        cv.waitKey()

rescale_and_show_random()