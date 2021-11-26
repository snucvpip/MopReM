import os
import time

import cv2
import numpy as np
from PIL import Image

# import pre_processing
import epll.denoise
import post_process.post_process as post_process

pardir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


class MopReM:
    def __init__(self, n):
        self.n = n


    def pre_process(self):
        datadir = os.path.join(pardir, 'data', self.n)
        assert os.path.exists(datadir), print('Pre process: datadir not exists')
        resultdir = os.path.join(pardir, 'result', self.n)
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)
        
        start = time.time()
        # pre_process
        end = time.time()
        print('Post processing time: {:.1f}s'.format(end-start))


    def demoire(self):
        datadir = os.path.join(pardir, 'result/pre_process', self.n)
        assert os.path.exists(datadir), print('Demoire: datadir not exists')
        resultdir = os.path.join(pardir, 'result/demoire', self.n)
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)
        
        source = os.path.join(datadir, 'source.png')
        target = os.path.join(datadir, 'target.png')

        # demoire
        start = time.time()
        epll.denoise.main(source, target, resultdir)
        end = time.time()
        print('Demoiring time: {:.1f}s'.format(end-start))


    def post_process(self):
        datadir = os.path.join(pardir, 'result/demoire', self.n)
        assert os.path.exists(datadir), print('Post process: datadir not exists')
        resultdir = os.path.join(pardir, 'result/post_process', self.n)
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        source = os.path.join(datadir, 'source')
        target = os.path.join(datadir, 'target')

        start = time.time()

        # post_process
        im = cv2.imread(source, cv2.IMREAD_COLOR)
        imReference = cv2.imread(target, cv2.IMREAD_COLOR)

        src, tar = post_process.CropImage(im, imReference)
        mse = post_process.mean_squared_error(src, tar)
        psnr = post_process.peak_signal_noise_ratio(src, tar)
        ssim, diff = post_process.structural_similarity(src, tar, multichannel=True, full=True)

        diff = (diff * 255).astype("uint8")
        diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

        fig, ax = post_process.plt.subplot(ncols=4, figsize=(15, 5))
        ax[0].imshow(src), ax[0].set_title('Source')
        ax[1].imshow(tar), ax[1].set_title('Target')
        ax[2].imshow(np.abs(src - tar)), ax[2].set_title(f'MSE:{round(mse, 2)}, PSNR:{round(psnr, 2)}, SSIM:{round(ssim, 2)}')
        ax[3].imshow(diff, cmap='gray', vmin=0, vmax=255), ax[3].set_title('Difference')
        post_process.plt.show()

        cv2.imwrite(os.path.join(resultdir, "source_cropped.png"), src)
        cv2.imwrite(os.path.join(resultdir, "target_cropped.png"), tar)

        end = time.time()
        print('Post processing time: {:.1f}s'.format(end-start))


if __name__ == '__main__':
    datadir = os.path.join(pardir, 'data')
    childlist = next(os.walk(datadir))[1]
    childlist.sort()
    print('total data number: ', len(childlist))

    start = time.time()
    for n in childlist:
        print('{} data demoireing...'.format(n))
        loop_start = time.time()
        model = MopReM(n)
        model.pre_process()
        model.demoire()
        model.post_process()
        loop_end = time.time()
        print('{} data demoireing time: {}s'.format(n, loop_end-loop_start))
    end = time.time()
    print('Total time: {}s'.format(end-start))
