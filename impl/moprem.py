import os
from re import M
import time
from functools import partial
from multiprocessing import Pool
import cv2
import numpy as np
from skimage.metrics import structural_similarity

import pre_process.pre_process as pre_process
import post_process.post_process as post_process
import post_process.eval as eval

import demoire.epll.denoise as epll
import demoire.remove_background.remove_background as remove_background

pardir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


class MopReM:
    def __init__(self):
        pass


    @classmethod
    def init_moprem(cls, pre_datadir, pre_resultdir,
                    epll_datadir, epll_resultdir,
                    post_datadir, post_resultdir,
                    eval_datadir, eval_resultdir,
                    remove_background_datadir, remove_background_resultdir):
        model = cls()
        model.pre_datadir                   = pre_datadir
        model.pre_resultdir                 = pre_resultdir
        model.epll_datadir                  = epll_datadir
        model.epll_resultdir                = epll_resultdir
        model.post_datadir                  = post_datadir
        model.post_resultdir                = post_resultdir
        model.eval_datadir                  = eval_datadir
        model.eval_resultdir                = eval_resultdir
        model.remove_background_datadir     = remove_background_datadir
        model.remove_background_resultdir   = remove_background_resultdir

        return model


    def pre(self):
        datadir = self.pre_datadir
        resultdir = self.pre_resultdir

        assert os.path.exists(datadir), print('Pre process: datadir not exists')

        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        source = os.path.join(datadir, 'source.png')
        target = os.path.join(datadir, 'target.png')
        print(resultdir)
        start = time.time()
        pre_process.main(source, target, resultdir)
        end = time.time()
        print('  Pre processing time\t\t\t+{:.2f}s'.format(end-start))


    def epll(self, background, moire, pooling=False):
        datadir = self.epll_datadir
        resultdir = self.epll_resultdir

        assert os.path.exists(datadir), print('EPLL: datadir not exists')

        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        files = next(os.walk(datadir))[2]
        files.sort()
        
        if pooling:
            targets = [os.path.join(datadir, f) for f in files]
            targets = targets[1:]

            start = time.time()
            pool = Pool(processes=5)
            # background seperation
            pool.map(partial(epll.main, matfile=background, DC=True, resultdir=resultdir), targets)
            # moire seperation
            pool.map(partial(epll.main, matfile=moire, DC=False, resultdir=resultdir), targets)
            pool.close()
            pool.join()
            end = time.time()

        else:
            start = time.time()
            for f in files:
                if not f == '1-1.png':
                    continue
                target = os.path.join(datadir, f)
                # background seperation
                epll.main(target, background, True, resultdir)
                # moire seperation
                epll.main(target, moire, False, resultdir)
            end = time.time()

        print('  Epll processing time\t\t\t+{:.2f}s'.format(end-start))

    
    def remove_background(self):
        datadir = self.remove_background_datadir
        resultdir = self.remove_background_resultdir

        assert os.path.exists(datadir), print('Remove background: datadir not exists')

        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        files = next(os.walk(datadir))[2]
        files.sort()

        start = time.time()
        for f in files:
            target = os.path.join(datadir, f)
            remove_background.main(target, resultdir)
        end = time.time()

        print('  Remove background processing time\t+{:.2f}s'.format(end-start))


    def post(self):
        datadir = self.post_datadir
        resultdir = self.post_resultdir

        assert os.path.exists(datadir), print('Demoire: datadir not exists')
        
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        # post-process
        start = time.time()
        post_process.main(pre_process.snapshot_info, datadir, resultdir)
        end = time.time()
        print('  Post processing time\t\t\t+{:.2f}s'.format(end-start))

        
    def eval(self):
        datadir = self.pre_datadir
        source = os.path.join(datadir, 'source.png')
        target = os.path.join(datadir, 'target.png')
        
        datadir = self.eval_datadir
        resultdir = self.eval_resultdir

        assert os.path.exists(datadir), print('Eval: datadir not exists')

        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

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

    print()
    print('MopReM : Moiré Pattern Removal for Mobile, Texts/Diagrams on Single-colored Background')
    print('PIP team, GilHo Roh, Sangjun Son, Jiwon Lee, Dongseok Choi')
    print('2021 fall SNU computer vision project')
    print()

    exclude = ['etc', '.ipynb_checkpoints', 'physics']
    datadir = os.path.join(pardir, 'data')
    imgs = [dirname for dirname in next(os.walk(datadir))[1] if not (dirname in exclude)]
    imgs.sort()

    print('\n# of images: ', len(imgs))

    start = time.time()
    for imgdir in imgs:
        print('\n--------------------------------------------------')
        print('\'{}\''.format(imgdir))

        resultdir = os.path.join(pardir, 'result', imgdir)

        pre_datadir = os.path.join(datadir, imgdir)
        pre_resultdir = os.path.join(resultdir, 'pre')

        epll_datadir = pre_resultdir
        epll_resultdir = os.path.join(resultdir, 'epll')

        remove_background_datadir = epll_resultdir
        remove_background_resultdir = os.path.join(resultdir, 'remove_background')

        post_datadir = epll_resultdir
        # post_datadir = remove_background_resultdir
        post_resultdir = os.path.join(resultdir, 'post')

        eval_datadir = epll_resultdir
        # eval_datadir = post_resultdir
        eval_resultdir = os.path.join(resultdir, 'eval')

        model = MopReM.init_moprem( pre_datadir, pre_resultdir, 
                                    epll_datadir, epll_resultdir, 
                                    post_datadir, post_resultdir,
                                    eval_datadir, eval_resultdir,
                                    remove_background_datadir, remove_background_resultdir )

        model.pre()

        model.epll(  background='GSModel_8x8_200_2M_noDC_zeromean.mat',
                     moire='GMM_8x8_200_1500.mat',
                     pooling=False  )

        # model.remove_background()

        # model.post()

        model.eval()

    end = time.time()
    print('==================================================')
    print('Total elapsed time\t\t\t{:.2f}s\n'.format(end-start))