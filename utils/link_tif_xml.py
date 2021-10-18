import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET


def extract_xml(fileId):
    tree = ET.parse('data/train_xml/'+fileId+'.xml')
    zone = tree.findall("{http://lamp.cfar.umd.edu/GEDI}DL_DOCUMENT/{http://lamp.cfar.umd.edu/GEDI}DL_PAGE/{http://lamp.cfar.umd.edu/GEDI}DL_ZONE")
    if len(zone) > 0:
        if zone[0].attrib['gedi_type'] == 'DLSignature':
            return 1
        else:
            return 0
    else:
        return 0


def dataframe_train():
    """
    Builds the dataframe having the true value of the target from .xml files.
    The target is the presence (target = 1) or absence (target = 0) of signature(s) in the .tif files.
    :return: a dataframe with 2 columns: id, signature_true (0 if none or 1 if at least one)
    """
    df = pd.DataFrame(columns=["Id", "signature_true"])
    for filename in os.listdir('data/train_xml/'):
        Id = os.path.basename(filename).split('.')[0]  # name of file
        res = extract_xml(fileId=Id)
        data = [Id, res]
        df.loc[len(df)] = data
    return df


if __name__ == '__main__':
    print(extract_xml('0a2c344efb5dd5b88450eec236a2aa3b_2'))
    print(extract_xml('0a948131fe85c38152c0b9b22f7c09fc_3'))
    #dataframe_train().to_csv('./data/df_train.csv', sep=',', index=False, mode='w')

