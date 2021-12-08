import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans

def dominantColors(img, n_clusters=3):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
    #reshaping to a list of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    #using k-means to cluster pixels
    kmeans = KMeans(n_clusters, max_iter=1)
    kmeans.fit(img)

    #the cluster centers are our dominant colors.
    return kmeans.cluster_centers_

def smoothing(img):
    pim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = pim.shape

    coordinates = np.array(peak_local_max(pim, min_distance=3))
    N = coordinates.shape[0]

    d = np.sqrt(width*height/N)
    z = int(d/2)

    im2 = img.copy().astype(np.uint32)
    for y, x in coordinates:
    #     if 100 < pim[y, x] < 150: continue
        im2[y-z:y+z, x-z:x+z] = (im2[y-z:y+z, x-z:x+z] + img[y, x]) / 2
      
    return im2.astype(np.uint8)
  
def seg(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    G = cv2.GaussianBlur(gray,(3,3),0.5)
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    ret, thresh = cv2.threshold(G,0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    I = np.array(rgb)
    x, y, z = I.shape
    I[thresh <= 254] = [255,255,255]                
    I2 = cv2.GaussianBlur(I, (3,3), 0.5) 
    return I2
    
def main(target, resultdir):
    img = cv2.imread(target)
    img = smoothing(img)   
    I_seg = seg(img)   
    assert os.path.exists(resultdir), print('result directory not exists')
    filename = os.path.basename(target)
    plt.imsave(os.path.join(resultdir, filename), I_seg)
    