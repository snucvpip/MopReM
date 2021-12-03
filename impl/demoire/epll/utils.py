'''
Implementation of matlab built-in functions & EPLL functions
'''
import numpy as np
from scipy.io import loadmat
import os


def get_gs_matrix(path, DC=False):
    '''
    .mat file loader\n
    only 'GSModel_8x8_200_2M_noDC_zeromean.mat' file load available.
    '''
    mat = loadmat(path)
    ret = {}

    # EPLL
    if DC:  
        GS = mat['GS']
        ret['dim']           = GS['dim'][0,0][0][0]
        ret['nmodels']       = GS['nmodels'][0,0][0][0]
        ret['means']         = GS['means'][0,0]
        ret['covs']          = GS['covs'][0,0]
        ret['invcovs']       = GS['invcovs'][0,0]
        ret['mixweights']    = GS['mixweights'][0,0]
    
    # NIPSGMM
    else:   
        GS = mat['GMM']
        ret['nmodels']       = GS['nmodels'][0,0][0][0]
        ret['mixweights']    = GS['mixweights'][0,0].T
        ret['covs']          = GS['covs'][0,0]
        ret['means']         = np.zeros((64, 200))

    return ret


def im2col (mtx, block_size):
    mtx_shape = mtx.shape
    sx = mtx_shape[0]-block_size[0]+1
    sy = mtx_shape[1]-block_size[1]+1

    # If the number of lines # Let A m × n, for the [PQ] of the block division, the final matrix of p × q, is the number of columns (m-p + 1) × (n-q + 1).
    result = np.empty((block_size[0]*block_size[1], sx*sy))

    # Moved along the line, so the first holding column (i) does not move down along the row (j)
    for i in range(sy):
        for j in range(sx):
            result[:,i*sx+j] = mtx[j:j+block_size[0],i:i+block_size[1]].ravel(order='F')

    return result


def accumarray(inds, data, out_shape, method='mean'):
    '''
    matlab accumarray implementation\n
    reference : https://kr.mathworks.com/help/matlab/ref/accumarray.html
    '''
    assert method == 'mean',    print('TypeError: only mean function is implemented')

    out = np.zeros(out_shape)
    counts = np.zeros(out_shape)

    for data_idx, group in enumerate(inds):
        out[group] += data[data_idx]
        counts[group] += 1

    out /= counts
    
    return out


def scol2im(Z, patchSize, mm, nn, method='average'):
    '''
    epll scol2im implementation\n
    only 'average' method available.
    '''
    method = method.lower()
    assert method == 'average', print('TypeError: only average scol2im is implemented')
    
    t = np.array(list(range(mm*nn))).reshape(nn, mm).T
    temp = im2col(t, (patchSize, patchSize)).astype('int64')
    I = accumarray(temp.T.flatten().tolist(), Z.T.flatten().tolist(), mm*nn, 'mean').reshape(nn, mm).T
    # I2 = np.bincount(temp.T.flatten(), weights=Z.T.flatten()).reshape(nn,mm).T
    return I