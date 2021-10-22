import os
import pandas as pd
from utils.opencv_test import contours_model
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2 as cv


model = keras.models.load_model('models/model2')
img_height = 168
img_width = 128
class_names = ['0', '1']

def dataframe_test():
    """
    Builds the dataframe having the true value of the target from .xml files.
    The target is the presence (target = 1) or absence (target = 0) of signature(s) in the .tif files.
    :return: a dataframe with 2 columns: id, signature_true (0 if none or 1 if at least one)
    """
    df = pd.DataFrame(columns=["Id", "Expected"])
    for filename in os.listdir('data/test/'):
        Id = os.path.basename(filename).split('.')[0]  # name of file
        img = tf.keras.utils.load_img(
            'data/test2/' + filename, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        res = class_names[np.argmax(score)]
        data = [Id, res]
        df.loc[len(df)] = data
    return df


def dataframe_test_page_width():
    """
    Builds the dataframe having the true value of the target from .xml files.
    The target is the presence (target = 1) or absence (target = 0) of signature(s) in the .tif files.
    :return: a dataframe with 2 columns: id, signature_true (0 if none or 1 if at least one)
    """
    df = pd.DataFrame(columns=["Id", "Expected"])
    for filename in os.listdir('data/test/'):
        Id = os.path.basename(filename).split('.')[0]  # name of file
        img = cv.imread('data/test/'+filename, -1)
        page_w = img.shape[1]
        res = 0
        if page_w < 2401:
            res = 1
        data = [Id, res]
        df.loc[len(df)] = data
    return df


if __name__ == '__main__':
    dataframe_test_page_width().to_csv('./data/df_test_page_width_20211022.csv', sep=',', index=False, mode='w')

