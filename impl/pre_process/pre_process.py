import numpy as np

from pre_process.pre_processing import *
import os
import cv2
import math
from skimage.feature import peak_local_max
from PIL import Image


def GetLimitNxNy(img):
    pim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = pim.shape
    n_pixel = 10
        
    coordinates = np.array(peak_local_max(pim, min_distance=3))
    N = coordinates.shape[0]
    
    d = np.ceil(np.sqrt(width*height/N))
    
    return tuple(np.array([width, height]) // int(d * n_pixel))


def Snapshot(img, resultdir, level=5):
    #################### EXPLANATION #####################
    # This function makes snapshots of each level
    # The level is equivalent of magnification (the higher level is, the more magnified image is)
    # The nx, ny determines dividing range
    # the cropped images in neighborhood have same area for post-processing
    ######################################################

    ##################### PARAMETERS #####################
    # level: division in range (1 ~ nx), (1 ~ ny)
    # nx: maximum number of dividing in x-axis
    # ny: maximum number of dividing in y-axis
    ######################################################

    ################### USAGE EXAMPLE ####################
    # Snapshot(5, 2, 2)
    # This usage will divide image in 5 levels.
    # The maximum number of dividing is 2 in x, y axis
    # nx = [1, 3, 5]
    # ny = [1, 3, 5]
    # level0 = original input image
    # level1 = 3 pieces in x-axis X 3 pieces in y-axis = 9 pieces
    # level2 = 5 pieces in x-axis X 5 pieces in y-axis = 25 pieces
    ######################################################

    snapshot_info = []
    nx, ny = GetLimitNxNy(img)
    list_nx = np.ceil(np.linspace(1, nx, level)).astype(int)
    list_ny = np.ceil(np.linspace(1, ny, level)).astype(int)

    dir_path = resultdir

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for i in range(level):

        width = int(math.ceil(img.shape[1] * 2 / (list_nx[i] + 1)))
        height = int(math.ceil(img.shape[0] * 2 / (list_ny[i] + 1)))

        num_image = 1
        stride = (math.floor(width / 2), math.floor(height / 2))
        y_start, y_end = (0, height)

        for y in range(list_ny[i]):
            x_start, x_end = (0, width)

            for x in range(list_nx[i]):
                cv2.imwrite(dir_path + '/' + str(i) + '-' + str(num_image) + '.png', img[y_start:y_end, x_start:x_end])

                if list_nx[i] >= 2 and x == list_nx[i] - 2:
                    x_start = img.shape[1] - 1 - width
                    x_end = img.shape[1] - 1
                else:
                    x_start += stride[0]
                    x_end += stride[0]

                num_image += 1

            if list_ny[i] >= 2 and y == list_ny[i] - 2:
                y_start = img.shape[0] - 1 - height
                y_end = img.shape[0] - 1
            else:
                y_start += stride[1]
                y_end += stride[1]

        snapshot_info.append(((list_nx[i], list_ny[i]), (width, height)))
        print('level{} Complete, nx: {}, ny: {}, width: {}, height: {}'.format(i, list_nx[i], list_ny[i], width, height))

    return snapshot_info


def main(source, target, resultdir):
    img = np.array(Image.open(source))
    Snapshot(img, resultdir)

