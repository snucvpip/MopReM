import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, color, morphology
from skimage.segmentation import flood, flood_fill
import cv2 as cv

def flood_fill_white(img):
    I = np.array(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    rgb = np.transpose(I, (2,0,1))
    RGB = rgb
    R, G, B = RGB[0], RGB[1], RGB[2]

    # Fill a square near the middle with value 127, starting at index (76, 76)
    Rf = flood_fill(R, (400, 500), 255, tolerance=70)
    Gf = flood_fill(G, (400, 500), 255, tolerance=70)
    Bf = flood_fill(B, (400, 500), 255, tolerance=70)

    new_I = np.array([Rf,Gf,Bf])
    FF = np.transpose(new_I,(1,2,0))
    
    return FF
    
if __name__ == "__main__":
    img = cv.imread('Images/smaple.jpg')

    FF = flood_fill_white(img)

    plt.axis('off')
    plt.imshow(FF)

    plt.show()