import numpy as np

from pre_processing import *
import os
import math
from skimage.feature import peak_local_max

dirName = "data/sample"
imFilename = os.path.join(dirName, "source.jpg")
image = cv2.imread(imFilename)


def GetLimitNxNy(img):
    pim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = pim.shape
    n_pixel = 10
        
    coordinates = np.array(peak_local_max(pim, min_distance=3))
    N = coordinates.shape[0]
    
    d = np.ceil(np.sqrt(width*height/N))
    
    return tuple(np.array([width, height]) // int(d * n_pixel))


def Snapshot(img, level=10):
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
    print(nx, ny)
    list_nx = np.linspace(1, nx, level)
    list_ny = np.linspace(1, ny, level)

    for i in range(level):
        dir_path = dirName + '/snapshot_level' + str(i)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        width = int(math.ceil(image.shape[1] * 2 / (list_nx[i] + 1)))
        height = int(math.ceil(image.shape[0] * 2 / (list_ny[i] + 1)))

        num_image = 1
        stride = (math.floor(width / 2), math.floor(height / 2))
        y_start, y_end = (0, height)

        for y in range(int(math.ceil(list_ny[i]))):
            x_start, x_end = (0, width)

            for x in range(int(math.ceil(list_nx[i]))):
                cv2.imwrite(dir_path + '/' + str(num_image) + '.jpg', image[y_start:y_end, x_start:x_end])

                if list_nx[i] >= 2 and x == list_nx[i] - 2:
                    x_start = image.shape[1] - 1 - width
                    x_end = image.shape[1] - 1
                else:
                    x_start += stride[0]
                    x_end += stride[0]

                num_image += 1

            if list_ny[i] >= 2 and y == list_ny[i] - 2:
                y_start = image.shape[0] - 1 - height
                y_end = image.shape[0] - 1
            else:
                y_start += stride[1]
                y_end += stride[1]

        snapshot_info.append(((list_nx[i], list_ny[i]), (width, height)))
        print('level{} Complete, nx: {}, ny: {}, width: {}, height: {}'.format(i, list_nx[i], list_ny[i], width, height))

    return snapshot_info


if __name__ == "__main__":
    Snapshot(image)
