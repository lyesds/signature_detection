import os
import pandas as pd
from utils.opencv_test import contours_model


def dataframe_test():
    """
    Builds the dataframe having the true value of the target from .xml files.
    The target is the presence (target = 1) or absence (target = 0) of signature(s) in the .tif files.
    :return: a dataframe with 2 columns: id, signature_true (0 if none or 1 if at least one)
    """
    df = pd.DataFrame(columns=["Id", "Expected"])
    for filename in os.listdir('data/test/'):
        Id = os.path.basename(filename).split('.')[0]  # name of file
        res = contours_model(fileDir='data/test/', fileId=Id)
        data = [Id, res]
        df.loc[len(df)] = data
    return df


if __name__ == '__main__':
    dataframe_test().to_csv('./data/df_test_contours_model_20211019b.csv', sep=',', index=False, mode='w')

