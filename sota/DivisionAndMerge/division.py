import cv2
import glob
import math
import numpy as np
import matplotlib.pyplot as plt


datadir = './data'
resultdir = './image'


def main():
    for img_path in glob.glob(datadir + '/1-1.jpg'):
        img = cv2.imread(img_path)

        # image cutting
        img = np.array(img)
        size_x = img.shape[1]
        size_y = img.shape[0]
        partial_size = 40
        idx = 1
        
        for i in range(partial_size):
            for j in range(partial_size):
                partial_img = img[math.floor(i/partial_size*size_y):math.floor((i+1)/partial_size*size_y), math.floor(j/partial_size*size_x):math.floor((j+1)/partial_size*size_x), :]
                partial_img = cv2.GaussianBlur(partial_img, (0, 0), 1.5)
                cv2.imwrite(resultdir + "/" + img_path[img_path.find(".jpg") - 3:img_path.find(".jpg")] + "_part" + str(idx) + ".jpg", partial_img)
                idx = idx + 1 

if __name__ == "__main__":
    main()
