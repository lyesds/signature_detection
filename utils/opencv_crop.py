import cv2 as cv
import glob
import random
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from utils.xml_details import extract_xml_details
from sklearn import metrics


train = glob.glob('data/train/*.tif')
test = glob.glob('data/test/*.tif')


def crop_relative(subset, nsample):
    pool = random.choices(subset, k=nsample)
    # pool = ['data/test\\be3ac3513667c46886e2cf27ddedbe36_2.tif', 'data/test\\51e2bcf8a78d836c45bd5d41e4555331.tif', 'data/test\\afcfb90e729e30b3dc7b09353200dac3.tif']
    print(pool)
    for i in pool:
        img= cv.imread(i, -1)
        dim = (480, 630)
        # resize image
        resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        cv.imshow('resized', resized)
        cv.waitKey()

        page_h = img.shape[0]
        page_w = img.shape[1]
        crop_img = img[int(page_h*0.4):int(page_h*0.9), int(page_w*0.4):int(page_w*0.9)]
        cv.imshow("Cropped TO SIGNATURE AVERAGE LOCATION", crop_img)
        cv.waitKey()


def crop_exact(subset, nsample):
    #pool = random.choices(subset, k=nsample)
    #print(pool)
    #for i in pool:
    for i in train:
        Id = os.path.basename(i).split('.')[0]  # name of file
        res = extract_xml_details(fileId=Id)

        img= cv.imread(i, -1)

        if int(res[2]) > 0:
            crop_img = img[int(res[3]):int(res[3])+int(res[5]), int(res[2]):int(res[2])+int(res[4])]
            cv.imwrite('data/train3/1/' + Id + '.jpg', crop_img)
        else:
            y = img.shape[1]*0.75
            x = img.shape[0]*0.5
            crop_img = img[int(y):int(y)+256, int(x):int(x)+256]
            cv.imwrite('data/train3/0/' + Id + '.jpg', crop_img)
        #cv.imshow("Cropped to signature location or bottom right corner", crop_img)
        #cv.waitKey()


if __name__ == '__main__':
    # crop_relative(subset=test, nsample=3)
    crop_exact(subset=train, nsample=3)
