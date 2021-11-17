import math
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2

img_path = './image'
y = 3264
x = 2448
n = 40
dx = x//40
dy = y//40

image = np.zeros(shape=(y, x, 3))

def mosaic():
    num = 0
    for img in glob.glob(img_path+'/*.jpg'):
        str = img[16:-4]
        #print(str)
        num = int(str)-1
        I = cv2.imread(img)
        arr = np.array(I)
        #plt.imshow(arr)
        #plt.show()
        #print(arr.shape)
        start_x = num%n
        start_y = num//n
        insert_mosaic(start_x, start_y, arr)
        #num += 1              
            
def insert_mosaic(start_x, start_y, img):
    #print(start_x, start_y)
    sx = start_x*dx
    sy = start_y*dy
    #print(sx,sy)
    for i in range(dx):
        for j in range(dy):
            image[sy+j][sx+i] = img[j][i]
            #print(image[sy+j][sx+i])
            #print(img[j][i])    
    #plt.imshow(image.astype(np.uint8))
    #plt.show()

def main():
    #print(dx, dy)
    mosaic()
    cv2.imwrite("mosaic.jpg", image)

if __name__ == "__main__":
    main()