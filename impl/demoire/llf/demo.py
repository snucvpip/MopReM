import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import llf

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

#img = cv.imread('Images/1-1_denoise.jpg')
img = cv.imread('Images/smaple.jpg')
I = np.array(cv.cvtColor(img, cv.COLOR_BGR2RGB))
rgb = np.transpose(I, (2,0,1))
RGB = np.float32(rgb/255)
R, G, B = RGB[0], RGB[1], RGB[2]

sigma=0.4
N=4
fact=1

Rf = llf.llf(R,sigma, fact, N)
Gf = llf.llf(G,sigma, fact, N)
Bf = llf.llf(B,sigma, fact, N)

new_I = np.array([Rf,Gf,Bf])
LLF = np.transpose(new_I,(1,2,0))


plt.imshow(LLF)
plt.axis('off')
plt.show()



