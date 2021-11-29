import numpy as np
import os
from demoire.epll.utils import get_gs_matrix, im2col, scol2im
import math
import time

def loggausspdf2(X, sigma):
    '''
    log pdf of Gaussian with zero mena\n
    Based on code written by Mo Chen (mochen@ie.cuhk.edu.hk). March 2009.
    '''
    d = X.shape[0]

    R = np.linalg.cholesky(sigma).T     # same to matlab 'chol'

    tmp = np.dot(np.linalg.inv(R.T), X)
    q = np.sum(tmp*tmp, axis=0, keepdims=True)  # quadratic term (M distance)
    c = d*np.log(2*np.pi) + 2*sum(np.log(np.diag(R))); # normalization constant
    y = -(c+q)/2

    return y


def aprxMAPGMM(Y, patchSize, noiseSD, imsize, GS, excludeList=None, SigmaNoise=None):
    '''
    approximate GMM MAP estimation - a single iteration of the "hard version"
    EM MAP procedure (see paper for a reference)

    Inputs:
        Y - the noisy patches (in columns)\n
        noiseSD - noise standard deviation\n
        imsize - size of the original image (not used in this case, but may be
        used for non local priors)\n
        GS - the gaussian mixture model structure\n
        excludeList - used only for inpainting, misleading name - it's a list
        of patch indices to use for estimation, the rest are just ignored\n
        SigmaNoise - if the noise is non-white, this is the noise covariance
        matrix

    Outputs:
        Xhat - the restore patches
    '''
    # handle exclusion list - used for inpainting
    if not excludeList:
        excludeList = []
    
    # Supports general noise covariance matrices
    if not SigmaNoise:
        SigmaNoise = (noiseSD**2) * np.eye(patchSize**2)

    if len(excludeList) != 0:
        T = Y
        Y = Y[:, excludeList]

    # remove DC component
    meanY = np.array(np.mean(Y, axis=0)).reshape(1, Y.shape[1])
    Y = Y - meanY

    # calculate assignment probabilities for each mixture component for all
    # patches
    workingdir = os.path.dirname(os.path.realpath(__file__))
    datadir = os.path.join(workingdir, 'data')
    GS2 = get_gs_matrix(datadir)
    PYZ = np.zeros((GS['nmodels'],Y.shape[1]))
    for i in range(GS['nmodels']):
        GS2['covs'][:,:,i] = GS['covs'][:,:,i] + SigmaNoise
        PYZ[i,:] = np.log(GS['mixweights'][i]) + loggausspdf2(Y, GS2['covs'][:,:,i])

    ks = np.argmax(PYZ, axis=0).reshape(1, PYZ.shape[1])

    Xhat = np.zeros(Y.shape)
    for i in range(GS['nmodels']):
        inds = list(np.where(ks[0]==i))[0]
        Xhat[:,inds] = np.dot( np.linalg.inv(GS['covs'][:,:,i]+SigmaNoise), 
                    np.dot(GS['covs'][:,:,i], Y[:,inds]) + np.dot( SigmaNoise, np.tile(GS['means'][:,i].reshape(GS['means'].shape[0], 1),(1,len(inds))) ) )

    if excludeList:
        tt = T
        tt[:,excludeList] = Xhat + meanY
        Xhat = tt
    else:
        Xhat = Xhat + meanY

    return Xhat
    

def EPLLhalfQuadraticSplit(noiseI, rambda, patchSize, betas, T, I, LogLFunc, GS, excludeList=None, SigmaNoise=None):
    '''
    EPLLhalfQuadraticSplit - Minimizes the EPLL cost using half quadratic splitting
    as defined in the paper:
    "From Learning Models of Natural Image Patches to Whole Image Restoration"
    by Daniel Zoran and Yair Weiss, ICCV 2011

    Version 1.0 (21/10/2011)

    This function is for denoising and inpainting - for deblurring refer to
    EPLLhalfQuadraticSplitDeblur.m

    Inputs:
        noiseI - the noisy image
        lambda - the parameter lambda from Equation (2) in the paper (mostly
                used as the inverse of the noise variance. If a matrix is given, it
                should be the same size as the image (used for inpainting)
        patchSize - the size of patches to extract (single scalar, patches are
                    always square)
        betas - a list (1xM vector) of beta values, if the values are positive, they will be
                used as is, negative values will be ignored and beta will be estimated
                automatically from the noisy image (for as many iterations as there are
                in betas)
        T - The number of iterations to optimizie for X and Z at each beta
            value
        prior - a function handle to a function which calculates a MAP estimate
                using a given prior for a noisy patch at noise level beta, see examples in the
                demos
        I - the original image I, used only for PSNR calculations and
            comparisons
        LogLFunc - a function handle to calculate the log likelihood of patches
                    in the image, used for calculating the total cost (optional).

    Outputs:
        cleanI - the restored image
        psnr - a list of the psnr values obtained for each beta and iteration
        cost - if LogLFunc is given then this is the cost from Equation 2 in
                the paper at each value of beta.
    
    See demos in this same code package for examples on how to use this
    function for denoising and inpainting using some example priors
    (including the GMM prior used in the paper).

    All rights reserved to the authors of the paper (Daniel Zoran and Yair
    Weiss). If you have any questions, comments or suggestions please contact
    Daniel Zoran at daniez@cs.huji.ac.il.
    '''
    RealNoiseSD = np.sqrt(1/(rambda/patchSize**2))
    cost = []
    beta = np.abs(betas[0]/4)
    cleanI = noiseI
    # psnr = []

    for betaa in betas:
        loop_start = time.time()

        assert betaa >= 0, print('betaa should be positive')
        beta = betaa    

        for _ in range(T):
            Z = im2col(cleanI, (patchSize, patchSize))

            cleanZ = aprxMAPGMM(Z, patchSize, 1/np.sqrt(beta), noiseI.shape, GS, excludeList, SigmaNoise)

            I1 = scol2im(cleanZ, patchSize, noiseI.shape[0], noiseI.shape[1], 'average')

            counts = patchSize**2

            cleanI = noiseI*rambda / (rambda + beta*counts) + (beta*counts / (rambda + beta*counts)) * I1

            # psnr.append( 20 * math.log10(1/np.std(cleanI-I)) )

            # print('PSNR is:{:.2f} I1 PSNR:{:.2f}'.format(psnr[-1], 20*math.log10(1/np.std(I1-I))))
        
        loop_end = time.time()
        print('loop time elapsed: {:.2f}'.format(loop_end-loop_start))
    
    cleanI = cleanI.reshape(noiseI.shape)

    for i in range(cleanI.shape[0]):
        for j in range(cleanI.shape[1]):
            cleanI[i][j] = 1 if cleanI[i][j] > 1 else cleanI[i][j]
            cleanI[i][j] = 0 if cleanI[i][j] < 0 else cleanI[i][j]

    return cleanI, psnr, cost