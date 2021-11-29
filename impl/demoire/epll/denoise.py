import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import os

from demoire.epll.epll import EPLLhalfQuadraticSplit
from demoire.epll.utils import get_gs_matrix


def process(noiseI, GS):
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
                                    I           = None, 
                                    LogLFunc    = LogLFunc, 
                                    GS          = GS,
                                    excludeList = None,
                                    SigmaNoise  = None 
                                )

    return cleanI


def denoise( target,
             datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'),
             convert_type = 'RGB' ):
    convert_type = convert_type.upper()
    GS = get_gs_matrix(datadir) 

    if convert_type == 'L':
        # sourceI = np.array(Image.open(source).convert(convert_type))/255
        targetI = np.array(Image.open(target).convert(convert_type))/255
        
        print('grayscale image denoising...')
        cleanI = process(targetI, GS)

    elif convert_type == 'RGB':
        # sourceI = np.array(Image.open(source).convert(convert_type))/255
        targetI = np.array(Image.open(target).convert(convert_type))/255
        cleanI = np.empty(targetI.shape)

        for i in range(3):
            print()
            if i == 0:
                print('R channel denoising...')
            elif i == 1:
                print('G channel denoising...')
            else :
                print('B channel denoising...')

            cleanI[:,:,i] = process(targetI[:,:,i], GS)
            
    else:
        print('ValueError: covert type should be grayscale(L) or RGB')
        exit(-1)

    return cleanI


def get_result(cleanI, resultpath):
    assert os.path.exists(os.path.dirname(resultpath)), print('result directory not exists')

    if cleanI.ndim == 2:
        cmap='gray'
    elif cleanI.ndim == 3:
        cmap=None
    else:
        print('image dimesion should be 2 or 3')
        exit(-1)

    plt.imsave(resultpath, cleanI)


def main(target, resultdir):
    cleanI = np.array(Image.open(target).convert('RGB'))/255
    # cleanI = denoise(target=target, convert_type='RGB')
    filename = os.path.basename(target)
    resultpath = os.path.join(resultdir, filename)
    get_result(cleanI, resultpath)