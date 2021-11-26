import cv2
import numpy as np
from skimage.segmentation import flood
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

def ExtractFrame(img, threshold=100, tolerance=100):
    pim = (img < threshold)*(255-img)
    pim = cv2.cvtColor(pim, cv2.COLOR_BGR2GRAY)

    visit = np.ones(pim.shape, dtype='uint8')
    result = np.zeros(pim.shape, dtype='uint8')
    for point in np.transpose(pim.nonzero())[:2]:
        point = tuple(point)
        if (visit[point] == False): continue

        pim *= visit
        mask = flood(pim, point, tolerance=tolerance)
        visit *= (mask != True)

        if (result.sum() < mask.sum()):
            result = mask
    return result
  
def OuterFrame(img, threshold=100, tolerance=100):
    pim = (img < threshold)*(255-img)
    pim = cv2.cvtColor(pim, cv2.COLOR_BGR2GRAY)

    result = np.zeros(pim.shape, dtype='uint8')
    nnz = np.array([[0, 0],
                    [0, pim.shape[1]-1],
                    [pim.shape[0]-1, 0],
                    [pim.shape[0]-1, pim.shape[1]-1]])
    for point in nnz:
        point = tuple(point)
        mask = flood(pim, point, tolerance=tolerance)
        result += mask
        
    return flood(result, (pim.shape[0]//2, pim.shape[1]//2))
        
def InnerFrame(img, threshold=100, tolerance=100):
    pim = (img < threshold)*(255-img)
    pim = cv2.cvtColor(pim, cv2.COLOR_BGR2GRAY)

    result = np.zeros(pim.shape, dtype='uint8')
    point = (pim.shape[0]//2, pim.shape[1]//2)        
    return flood(pim, point, tolerance=tolerance)

def Frame(image, tolerance=100):
    result = np.array(image)
    mask = OuterFrame(image, tolerance)
    result[mask == 0] = 255
    mask = InnerFrame(result, tolerance)
    result[mask] = 255
    return ExtractFrame(result, tolerance)*255

def Boundaries(pim):
    center = [pim.shape[0]//2, pim.shape[1]//2]
    mask = flood(pim, tuple(center))
    center = center[::-1]

    contours, hierarchy = cv2.findContours(mask*255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    dist = 0
    v1 = center
    for contour in contours:
        for coord in contour:
            curDist = np.linalg.norm(coord[0]-center)
            if dist < curDist:
                dist = curDist
                v1 = coord[0]

    dist = 0
    v2 = v1
    for contour in contours:
        for coord in contour:
            curDist = np.linalg.norm(coord[0]-v1)
            if dist < curDist:
                dist = curDist
                v2 = coord[0]

    dist = 0
    v3 = center
    for contour in contours:
        for coord in contour:
            curDist = np.linalg.norm(np.cross(v2-v1, v1-coord[0]))/np.linalg.norm(v2-v1)
            if dist < curDist:
                dist = curDist
                v3 = coord[0]

    dist = 0
    v4 = v3
    for contour in contours:
        for coord in contour:
            curDist = np.linalg.norm(coord[0]-v3)
            if dist < curDist:
                dist = curDist
                v4 = coord[0]
    
    v = np.array([v1, v2, v3, v4])
    vertices = np.zeros(v.shape, dtype=int)
    for i, c in enumerate(v):
        if (c < center).all():
            vertices[0] = c
            v = np.delete(v, i, 0)
            
    for i, c in enumerate(v):
        if (c > center).all():
            vertices[2] = c
            v = np.delete(v, i, 0)
    
    if (v[0,0] < v[1,0]):
        vertices[1] = v[1]
        vertices[3] = v[0]
    else:
        vertices[1] = v[0]
        vertices[3] = v[1]
    return vertices

def CropImage(im, imReference, tolerance=70):
    fim = Frame(im, tolerance=tolerance)
    vim = Boundaries(fim)
    print(vim)

    fimReference = Frame(imReference)
    vimReference = Boundaries(fimReference)
    print(vimReference)
    R = np.array([np.min(vimReference, 0), np.max(vimReference, 0)])
    
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].imshow(im)
    ax[0].scatter(vim[:,0], vim[:,1], marker='x', c='red', s=200, linewidth=3)

    ax[1].imshow(imReference)
    ax[1].scatter(vimReference[:,0], vimReference[:,1], marker='x', c='red', s=200, linewidth=3)
    plt.tight_layout()
    plt.savefig('images_to_crop.png', dpi=300)    

    H, _ = cv2.findHomography(vim, vimReference)
    height, width, channels = imReference.shape
    result = cv2.warpPerspective(im, H, (width, height))

    src =      result[R[0,1]:R[1,1], R[0,0]:R[1,0]]
    tar = imReference[R[0,1]:R[1,1], R[0,0]:R[1,0]]
    return src, tar