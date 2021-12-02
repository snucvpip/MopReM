import numpy as np
import os
from epll.utils import get_gs_matrix, im2col, scol2im
import math
import time

def loggausspdf2(X, sigma):
    d = X.shape[0]

    R = np.linalg.cholesky(sigma).T     # same to matlab 'chol'

    tmp = np.dot(np.linalg.inv(R.T), X)
    q = np.sum(tmp*tmp, axis=0, keepdims=True)  # quadratic term (M distance)
    c = d*np.log(2*np.pi) + 2*sum(np.log(np.diag(R))); # normalization constant
    y = -(c+q)/2

    return y


def aprxMAPGMM(Y, patchSize, noiseSD, imsize, GS, excludeList=None, SigmaNoise=None, DC=False):
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
    GS2 = get_gs_matrix(datadir, 'GMM_8x8_200_1500.mat', DC)
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
    

def EPLLhalfQuadraticSplit(noiseI, rambda, patchSize, betas, T, I, LogLFunc, GS, excludeList=None, SigmaNoise=None, DC=False):
    RealNoiseSD = np.sqrt(1/(rambda/patchSize**2))
    cost = []
    beta = np.abs(betas[0]/4)
    cleanI = noiseI
    psnr = []

    counter = 0
    for betaa in betas:
        loop_start = time.time()

        assert betaa >= 0, print('betaa should be positive')
        beta = betaa    

        for _ in range(T):
            Z = im2col(cleanI, (patchSize, patchSize))

            cleanZ = aprxMAPGMM(Z, patchSize, 1/np.sqrt(beta), noiseI.shape, GS, excludeList, SigmaNoise, DC)

            I1 = scol2im(cleanZ, patchSize, noiseI.shape[0], noiseI.shape[1], 'average')

            counts = patchSize**2

            cleanI = noiseI*rambda / (rambda + beta*counts) + (beta*counts / (rambda + beta*counts)) * I1

            # psnr.append( 20 * math.log10(1/np.std(cleanI-I)) )

            # print('PSNR is:{:.2f} I1 PSNR:{:.2f}'.format(psnr[-1], 20*math.log10(1/np.std(I1-I))))
        
        loop_end = time.time()
        counter += 1
        print('loop time elapsed: {:.1f}s ({}/{})'.format(loop_end-loop_start, str(counter), str(len(betas))))

    cleanI = cleanI.reshape(noiseI.shape)

    for i in range(cleanI.shape[0]):
        for j in range(cleanI.shape[1]):
            cleanI[i][j] = 1 if cleanI[i][j] > 1 else cleanI[i][j]
            cleanI[i][j] = 0 if cleanI[i][j] < 0 else cleanI[i][j]

    return cleanI, psnr, cost