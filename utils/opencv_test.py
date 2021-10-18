import cv2 as cv
import glob
import random
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from utils.link_tif_xml import extract_xml
from sklearn import metrics


train = glob.glob('data/train/*.tif')
test = glob.glob('data/test/*.tif')


k=3
#print(random.choices(train, k=k), random.choices(test, k=k))

def rescale_and_show_random():
    for i in random.choices(train, k=k):
        img= cv.imread(i, -1)
        dim = (480, 630)
        # resize image
        resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        cv.imshow('resized', resized)
        cv.waitKey()

        ret, thresh1 = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)
        plt.imshow(thresh1, 'gray')
        #plt.show()

        # dilate
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv.dilate(thresh1, kernel, iterations=5)
        plt.imshow(dilate, 'gray')
        #plt.show()

        contours, hierarchy = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        #print(len(contours))

        white_bg = 255 * np.ones_like(img)
        logic_list = []
        for i, c in enumerate(contours):
            #x, y, MA, ma, angle = cv.fitEllipse(c)
            area = cv.contourArea(c)
            x, y, w, h = cv.boundingRect(c)
            rect_area = w * h
            extent = float(area) / rect_area
            #if (y+h) > dilate.shape[1]/2 and w > dilate.shape[0]*0.1 and w/h < 5 and w/h > 1:
            if w > dilate.shape[0] * 0.1 and w / h < 5 and w / h > 1:
                #print(c.shape, extent, .2<extent<.35)
                logic_list.append(.2<extent<.35)
                cv.drawContours(white_bg, contours, i, 2, 2, cv.LINE_8, hierarchy, 0)
        #print(any(logic_list))
        # plt.axis('off')
        plt.imshow(white_bg)
        #plt.show()
        Expected = 0
        if any(logic_list):
            Expected = 1
        print(Expected)
        # return Expected



def contours_model(fileId, fileDir = 'data/train/'):
    img= cv.imread(fileDir+fileId+'.tif', -1)
    dim = (480, 630)
    # resize image
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    ret, thresh1 = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)
    # dilate
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv.dilate(thresh1, kernel, iterations=5)

    # contours
    contours, hierarchy = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    logic_list = []
    for i, c in enumerate(contours):
        area = cv.contourArea(c)
        x, y, w, h = cv.boundingRect(c)
        rect_area = w * h
        extent = float(area) / rect_area
        if (y+h) > dilate.shape[1]/2 and w > dilate.shape[0] * 0.05 and w / h < 5 and w / h > 1:
            logic_list.append(extent < .5)
    expected = 0
    if any(logic_list):
        expected = 1
    return expected


def dataframe_train():
    """
    Builds the dataframe having the pred value of the target from .tif files.
    The target is the presence (target = 1) or absence (target = 0) of signature(s) in the .tif files.
    :return: a dataframe with 2 columns: id, Expected (0 if none or 1 if at least one)
    """
    df = pd.DataFrame(columns=["Id", "y_true", "y_pred"])
    for filename in os.listdir('data/train/'):
        Id = os.path.basename(filename).split('.')[0]  # name of file
        res1 = extract_xml(fileId=Id)*1.0
        res2 = contours_model(fileDir='data/train/', fileId=Id)*1.0
        data = [Id, res1, res2]
        df.loc[len(df)] = data
    confusion_matrix = metrics.confusion_matrix(df['y_true'], df['y_pred'])
    print(confusion_matrix)
    return df


if __name__ == '__main__':
    # rescale_and_show_random()
    # print(rescale_and_show_random())
    print(dataframe_train().head(10))
