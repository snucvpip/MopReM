import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv

def seg(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    G = cv.GaussianBlur(gray,(3,3),0.5)
    rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    ret, thresh = cv.threshold(G,0,255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    I = np.array(rgb)
    x, y, z = I.shape
    for i in range(x):
        for j in range(y):
            if thresh[i][j] <=254:
                I[i][j] = [255,255,255]
                
    I2 = cv.GaussianBlur(I, (3,3), 0.5) 
    return I2
    
def main(target, resultdir):
    img = cv.imread(target)
    I = seg(img)   
    assert os.path.exists(resultdir), print('result directory not exists')
    filename = os.path.basename(target)
    plt.imsave(os.path.join(resultdir,filename), I)
    