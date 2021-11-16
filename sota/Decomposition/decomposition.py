
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import LLF
import convert

#LDPC how?
def decomposition(img):
    output = 0
    
    downsample = img
    down = []
    
    #decomposition
    for i in range(4):
        downsample = cv2.pyrDown(downsample)
        down.append(downsample)
    
    return output

def reconstruction(list):
    #how reconstrct dowm sampling imgae?
    print('reconstruction')
    return 0
    
def layer_decomposition():
    #use EPLL
    #make EPLL code to python
    print('divide')
    return 0

def LDPC(img):
    
    #poly = decomposition(img)
    ldpc_poly = []
    
    #for i in poly:
        #maybe use matlab code
    #    ld = layer_decomposition(i)
    #    ldpc_poly.append(ld)
        
    #ldpc = reconstruction(ldpc_poly)
    
    return img
    
if __name__ == '__main__':
    #img_moire = cv2.imread('moire.jpg')
    img_moire = Image.open('moire.jpg').convert('RGB')
    img = np.array(img_moire)
    
    rgb = np.transpose(img, (2,0,1))
    r, g, b = rgb[0], rgb[1], rgb[2]
    
    y, u, v = convert.RGB2YUV(r, g, b)
    
    #get Y channel
    Y = LDPC(y)

    rr, gg, bb = convert.YUV2RGB(Y, u, v)
    
    R = LDPC(rr)
    G = LDPC(gg)
    B = LDPC(bb)
    
    new_img = np.array([R,G,B])
    demoire = np.transpose(new_img, (1,2,0))
    
    plt.imshow(demoire.astype(np.uint8))
    plt.show()    
    
    