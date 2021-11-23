import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import os

from epll import *


def process(I, noiseI, GS):
    patchSize = 8
    noiseSD = 25/255
    
    # same to matlab seed
    # np.random.seed(1)
    # rand = np.array(norm.ppf(np.random.rand(I.shape[1], I.shape[0]))).T
    # noiseI = I + noiseSD * rand
    excludeList = []
    LogLFunc = []

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

    return cleanI


def denoise( datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'),
             target = 'cat.jpg',
             source = '', 
             convert_type = 'L' ):
    GS = get_gs_matrix(datadir) 

    if convert_type == 'L':
        sourceI = np.array(Image.open(os.path.join(datadir, source)).convert(convert_type))/255
        targetI = np.array(Image.open(os.path.join(datadir, target)).convert(convert_type))/255
        
        print('grayscale image denoising...')
        cleanI = process(targetI, sourceI, GS)

    elif convert_type == 'RGB':
        sourceI = np.array(Image.open(os.path.join(datadir, source)).convert(convert_type))/255
        targetI = np.array(Image.open(os.path.join(datadir, target)).convert(convert_type))/255
        cleanI = np.empty(I.shape)

        for i in range(3):
            if i == 0:
                print('R channel denoising...')
            elif i == 1:
                print('G channel denoising...')
            else :
                print('B channel denoising...')

            cleanI[:,:,i] = process(targetI[:,:,i], sourceI[:,:,i], GS)
            print()

    else:
        print('ValueError: covert type should be grayscale(L) or RGB')
        exit(-1)

    return I, noiseI, cleanI


def get_result(I, noiseI, cleanI, filename='result.png', show=False):
    resultdir = os.path.join( os.path.dirname(os.path.realpath(__file__)), 'result' )
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    if I.ndim == 2:
        cmap='gray'
    elif I.ndim == 3:
        cmap=None
    else:
        print('image dimesion should be 2 or 3')
        exit(-1)

    _, ax = plt.subplots(ncols=3, figsize=(15, 5))
    ax[0].imshow(I, cmap=cmap)
    ax[0].set_title('Original')
    ax[1].imshow(noiseI, cmap=cmap)
    ax[1].set_title('Corrupted Image')
    ax[2].imshow(cleanI, cmap=cmap)
    ax[2].set_title('Cleaned Image')
    plt.savefig(os.path.join(resultdir,filename), dpi=300)
    if show:
        plt.show()


if __name__ == '__main__':
    start = time.time()
    I, noiseI, cleanI = denoise(source='source_cropped.png', target='target_cropped.png', convert_type='L')
    end = time.time()
    print('time elapsed : {:.2f}'.format(end-start))

    get_result(I, noiseI, cleanI)