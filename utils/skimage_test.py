# some ideas come from here https://github.com/ahmetozlu/signature_extractor

# import os
import cv2 as cv
from skimage import measure
import matplotlib.pyplot as plt
import glob
import random


train = glob.glob('data/train/*.tif')
test = glob.glob('data/test/*.tif')

def find_blobs(subset, nsample):
    pool = random.choices(subset, k=nsample)
    print(pool)
    for i in pool:
        img = cv.imread(i, -1)

        # what = img > img.mean() # this convert to array with true / false values

        blobs_labels = measure.label(img, background=1)
        blobs_props = measure.regionprops(blobs_labels)
        #for blob in blobs_props:
        #     print(blob.area)
        giants = [x for x in blobs_props if x.area > 3000]
        bboxes = [x.bbox for x in giants]
        areas = [x.area for x in giants]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img)
        for box in bboxes:
            # ymin, xmin, a, ymax, xmax, b = box
            ymin, xmin,ymax, xmax = box
            if ymax - ymin < 25 or ymax - ymin > 250 or ymax == img.shape[0]:
                continue
            by = (ymin, ymax, ymax, ymin, ymin)
            bx = (xmin, xmin, xmax, xmax, xmin)
            ax.plot(bx, by, '-b', linewidth=2.5)

        ax.set_axis_off()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    find_blobs(subset=test, nsample=3)
