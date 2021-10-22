import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET


def extract_xml_details(fileId):
    tree = ET.parse('data/train_xml/'+fileId+'.xml')
    zonePage = tree.findall("{http://lamp.cfar.umd.edu/GEDI}DL_DOCUMENT/{http://lamp.cfar.umd.edu/GEDI}DL_PAGE")
    zone = tree.findall("{http://lamp.cfar.umd.edu/GEDI}DL_DOCUMENT/{http://lamp.cfar.umd.edu/GEDI}DL_PAGE/{http://lamp.cfar.umd.edu/GEDI}DL_ZONE")
    if len(zone) > 0:
        if zone[0].attrib['gedi_type'] == 'DLSignature':
            return zonePage[0].attrib['width'], zonePage[0].attrib['height'], zone[0].attrib['col'], zone[0].attrib['row'], zone[0].attrib['width'], zone[0].attrib['height']
        else:
            return zonePage[0].attrib['width'], zonePage[0].attrib['height'], 0, 0, 0, 0
    else:
        return zonePage[0].attrib['width'], zonePage[0].attrib['height'], 0, 0, 0, 0


def dataframe_train():
    """
    Builds the dataframe having the true value of the target from .xml files.
    The target is the presence (target = 1) or absence (target = 0) of signature(s) in the .tif files.
    :return: a dataframe with 2 columns: id, signature_true (0 if none or 1 if at least one)
    """
    df = pd.DataFrame(columns=["Id", "page_w", "page_h", "sign_x", "sign_y", "sign_w", "sign_h"])
    for filename in os.listdir('data/train_xml/'):
        Id = os.path.basename(filename).split('.')[0]  # name of file
        res = extract_xml_details(fileId=Id)
        data = [Id, int(res[0]), int(res[1]), int(res[2]), int(res[3]), int(res[4]), int(res[5])]
        df.loc[len(df)] = data
    return df

import tabulate





if __name__ == '__main__':
    '''print(extract_xml_details('0a2c344efb5dd5b88450eec236a2aa3b_2'))
    print(extract_xml_details('0a948131fe85c38152c0b9b22f7c09fc_3'))
    print(extract_xml_details('0a9aa4f1b00fb99492a54a99e70384be'))
    print(extract_xml_details('00ba5cc657c8c203c4ed5e339f7d50e9'))'''
    '''print(dataframe_train().head(10).to_markdown())
    dataframe_train().to_csv('./data/df_details_train.csv', sep=',', index=False, mode='w')'''
    df = pd.read_csv('data/df_details_train.csv')
    print(df.info())
    df_signed = df[df['sign_x'] > 0]
    print(df_signed.info())
    df_signed['relative_x'] = df_signed['sign_x'] / df_signed['page_w']
    df_signed['relative_y'] = df_signed['sign_y'] / df_signed['page_h']
    df_signed['relative_w'] = df_signed['sign_w'] / df_signed['page_w']
    df_signed['relative_h'] = df_signed['sign_h'] / df_signed['page_h']
    print(df_signed.relative_x.describe())
    print(df_signed.relative_y.describe())
    print(df_signed.relative_w.describe())
    print(df_signed.relative_h.describe())


