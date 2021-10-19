import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from utils.opencv_test import dataframe_train
import time


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
