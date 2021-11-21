import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import os

from epll import *

def denoise(datadir=os.path.join( os.path.dirname(os.path.realpath(__file__)), 'data' ),
            filename='160068.jpg'):
    patchSize = 8

    I = np.array(Image.open(os.path.join(datadir, filename)).convert('L'))/255

    noiseSD = 25/255
    
    # same to matlab seed
    np.random.seed(1)
    rand = np.array(norm.ppf(np.random.rand(I.shape[1], I.shape[0]))).T
    noiseI = I + noiseSD * rand
    excludeList = []
    LogLFunc = []

    GS = get_gs_matrix(datadir)

    cleanI, psnr, cost = EPLLhalfQuadraticSplit(
                                    noiseI      = noiseI, 
                                    rambda      = patchSize**2/noiseSD**2, 
                                    patchSize   = patchSize, 
                                    betas       = (1/(noiseSD**2))*np.array([1,4,8,16,32]), 
                                    T           = 1, 
                                    I           = I, 
                                    LogLFunc    = LogLFunc, 
                                    GS          = GS,
                                    excludeList = None,
                                    SigmaNoise  = None 
                                )

    return I, noiseI, cleanI


if __name__ == '__main__':
    start = time.time()
    I, noiseI, cleanI = denoise()
    end = time.time()
    print('time elapsed : {:.1f}'.format(end-start))

    resultdir = os.path.join( os.path.dirname(os.path.realpath(__file__)), 'result' )
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    ax[0].imshow(I, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(noiseI, cmap='gray')
    ax[1].set_title('Corrupted Image')
    ax[2].imshow(cleanI, cmap='gray')
    ax[2].set_title('Cleaned Image')
    plt.savefig(os.path.join(resultdir,'result.png'), dpi=300)
    # plt.show()