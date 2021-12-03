import numpy as np
import os
import cv2

def main(info, datadir, resultdir):
    level = len(info)
    img_width, img_height = info[0][1]

    for i in range(1, level):
        img_merge = np.empty(shape=(img_width, img_height, 3))
        nx, ny = info[i][0]
        partial_width, partial_height = info[i][1]
        stride = info[i][2]
        num = 1

        for y in range(ny):
            if y != ny - 1:
                for x in range(nx):
                    filename = str(i)+'-'+str(num)+'.png'
                    filepath = os.path.join(datadir, filename)
                    partial_image = cv2.imread(filepath,  cv2.IMREAD_COLOR)
                    if x == nx - 1:
                        img_merge[y * stride[1]: y * stride[1] + partial_height,\
                                    img_width-1-partial_width:img_width-1] = partial_image
                    else:
                        img_merge[y * stride[1]: y * stride[1] + partial_height,\
                                    x * stride[0]: x * stride[0] + partial_width] = partial_image
                    num += 1
            else:
                for x in range(nx):
                    filename = str(i)+'-'+str(num)+'.png'
                    filepath = os.path.join(datadir, filename)
                    partial_image = cv2.imread(filepath,  cv2.IMREAD_COLOR)
                    if x == nx - 1:
                        img_merge[img_height-1-partial_height:img_height-1,\
                        img_width-1-partial_width:img_width-1] = partial_image
                    else:
                        img_merge[img_height-1-partial_height:img_height-1,\
                        x * stride[0]: x * stride[0] + partial_width] = partial_image
                    num += 1

        result_name = 'clean'+str(i)+'.png'
        result_path = os.path.join(resultdir, result_name)
        cv2.imwrite(result_path, img_merge)