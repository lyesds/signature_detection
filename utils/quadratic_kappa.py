import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from utils.link_tif_xml import extract_xml
import time
import os

from tensorflow import keras
import tensorflow as tf
import numpy as np


model = keras.models.load_model('models/model3')
img_height = 168
img_width = 128
class_names = ['0', '1']

def dataframe_train():
    """
    Builds the dataframe having the pred value of the target from .tif files.
    The target is the presence (target = 1) or absence (target = 0) of signature(s) in the .tif files.
    :return: a dataframe with 2 columns: id, Expected (0 if none or 1 if at least one)
    """
    df = pd.DataFrame(columns=["Id", "y_true", "y_pred"])
    for filename in os.listdir('data/train_jpg/'):
        Id = os.path.basename(filename).split('.')[0]  # name of file
        res1 = extract_xml(fileId=Id)*1.0
        img = tf.keras.utils.load_img(
            'data/train_jpg/' + filename, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        res2 = int(class_names[np.argmax(score)])*1.0

        data = [Id, res1, res2]
        df.loc[len(df)] = data
    confus_matrix = confusion_matrix(df['y_true'], df['y_pred'])
    print(confus_matrix)
    return df


tic = time.perf_counter()
df = dataframe_train()
toc = time.perf_counter()
print(f"Built data frame Train in {toc - tic:0.4f} seconds")

print(df.shape)
print(round(cohen_kappa_score(df.y_true, df.y_pred, weights='quadratic'), 4))

# https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps#Rewrite-the-Quadratic-Kappa-Metric-function
def quadratic_kappa(actuals, preds, N=5):
    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values
    of adoption rating."""
    w = np.zeros((N, N))
    O = confusion_matrix(actuals, preds)
    for i in range(len(w)):
        for j in range(len(w)):
            w[i][j] = float(((i - j) ** 2) / (N - 1) ** 2)

    act_hist = np.zeros([N])
    for item in actuals:
        act_hist[item] += 1

    pred_hist = np.zeros([N])
    for item in preds:
        pred_hist[item] += 1

    E = np.outer(act_hist, pred_hist);
    E = E / E.sum();
    O = O / O.sum();

    num = 0
    den = 0
    for i in range(len(w)):
        for j in range(len(w)):
            num += w[i][j] * O[i][j]
            den += w[i][j] * E[i][j]
    return (1 - (num / den))


print(quadratic_kappa(actuals=df.y_true.to_numpy(dtype=int), preds=df.y_pred.to_numpy(dtype=int), N=2))

'''fp= 13; fn=20; a=109; b=142; x=116; y=135;
kappa = 1 - (251*(fp+fn)/(a*y+b*x))
print(kappa)'''
