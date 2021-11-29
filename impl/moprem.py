import os
import time

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

import pre_process.pre_process as pre_process
import post_process.post_process as post_process
import post_process.eval as eval

import demoire.epll.denoise 
import demoire.remove_background.remove_background as remove_background

pardir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


class MopReM:
    def __init__(self, n):
        self.n = n

    def pre(self):
        datadir = os.path.join(pardir, 'data', self.n)
        assert os.path.exists(datadir), print('Pre process: datadir not exists')
        resultdir = os.path.join(pardir, 'result/pre_process', self.n)
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        source = os.path.join(datadir, 'source.png')
        target = os.path.join(datadir, 'target.png')
        
        # pre-process
        start = time.time()
        pre_process.main(source, target, resultdir)
        end = time.time()
        print('  Pre processing time\t\t\t+{:.2f}s'.format(end-start))


    def demoire(self):
        datadir = os.path.join(pardir, 'result/pre_process', self.n)
        assert os.path.exists(datadir), print('Demoire: datadir not exists')
        files = next(os.walk(datadir))[2]
        files.sort()
        resultdir = os.path.join(pardir, 'result/demoire', self.n)
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        start = time.time()
        
#         # epll
#         for f in files:
#             target = os.path.join(datadir, f)
#             epll.denoise.main(target, resultdir)

        # remove_background
        for i in range(5):
            n = 0
            if i == 1:
                n = 16
            elif i ==2:
                n = 36
            elif i ==3:
                n = 81
            elif i ==3:
                n = 121
            for j in range(n):
                name = str(i)+'-'+str(j+1)+'.png'
                source = os.path.join(datadir, name)
                remove_background.main(source, resultdir)
                
        end = time.time()
        print('  Demoireing time\t\t\t+{:.2f}s'.format(end-start))


    def post(self):
        datadir = os.path.join(pardir, 'result/demoire', self.n)
        assert os.path.exists(datadir), print('Post process: datadir not exists')
        resultdir = os.path.join(pardir, 'result/post_process', self.n)
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        # post-process
        start = time.time()
        post_process.main(pre_process.snapshot_info, datadir, resultdir)
        end = time.time()
        print('  Post processing time\t\t\t+{:.2f}s'.format(end-start))

        
    def eval(self):
        datadir = os.path.join(pardir, 'data', self.n)
        assert os.path.exists(datadir), print('Eval: datadir not exists')
        resultdir = os.path.join(pardir, 'result/eval', self.n)
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        source = os.path.join(datadir, 'source.png')
        target = os.path.join(datadir, 'target.png')
        
        datadir = os.path.join(pardir, 'result/post_process', self.n)
        assert os.path.exists(datadir), print('Eval: datadir not exists')
        imReference = cv2.imread(target)
        files = next(os.walk(datadir))[2]
        files.sort()
        
        # eval
        start = time.time()
        
        best_clean = ""
        best_ssim = 0
        for f in files:
            clean = os.path.join(datadir, f)
            imClean = cv2.imread(clean)
            cle, tar = eval.CropImage(imClean, imReference)
            ssim = structural_similarity(cle, tar, multichannel=True)
            if best_ssim < ssim:
                best_ssim = ssim
                best_clean = clean
        print(best_clean)
        eval.main(source, target, best_clean, resultdir)
        end = time.time()
        print('  Eval time\t\t\t\t+{:.2f}s'.format(end-start))


if __name__ == '__main__':
    datadir = os.path.join(pardir, 'data')
    exclude = ['etc', '.ipynb_checkpoints', 'physics']
    childlist = [dirname for dirname in next(os.walk(datadir))[1] if not (dirname in exclude)]
    childlist.sort()
    print('\n# of images: ', len(childlist))

    start = time.time()
    for dirname in childlist:
        print('\n--------------------------------------------------')
        print('\'{}\''.format(dirname))
        loop_start = time.time()
        model = MopReM(dirname)
        model.pre()
        model.demoire()
        model.post()
        model.eval()
        loop_end = time.time()
        print('  Elapsed time\t\t\t\t{:.2f}s\n'.format(loop_end-loop_start))
    end = time.time()
    print('==================================================')
    print('Total elapsed time\t\t\t{:.2f}s\n'.format(end-start))
