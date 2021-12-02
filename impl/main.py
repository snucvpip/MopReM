import os
from re import M
import time
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from glob import glob
import cv2
import numpy as np

import epll.denoise
import pre_process.pre_process
import remove_background.remove_background
import post_process.post_process

pardir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


class MopReM:
    def __init__(self):
        pass


    @classmethod
    def init_moprem(cls, pre_datadir, pre_resultdir,
                    epll_datadir, epll_resultdir,
                    post_datadir, post_resultdir):
        model = MopReM()
        model.pre_datadir = pre_datadir
        model.pre_resultdir = pre_resultdir
        model.epll_datadir = epll_datadir
        model.epll_resultdir = epll_resultdir
        model.post_datadir = post_datadir
        model.post_resultdir = post_resultdir

        return model


    def pre_process(self):
        datadir = self.pre_datadir
        resultdir = self.pre_resultdir

        assert os.path.exists(datadir), print('Pre process: datadir not exists')

        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        source = os.path.join(datadir, 'source.png')
        target = os.path.join(datadir, 'target.png')
        print(resultdir)
        start = time.time()
        pre_process.pre_process.main(source, target, resultdir)
        end = time.time()
        print('Elapsed time : {:.1f}s\n'.format(end-start))


    def demoire(self, background, moire, pooling=False):
        datadir = self.epll_datadir
        resultdir = self.epll_resultdir

        assert os.path.exists(datadir), print('Demoire: datadir not exists')

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
            pool.map(partial(epll.denoise.main, matfile=background, DC=True, resultdir=resultdir), targets)
            # moire seperation
            pool.map(partial(epll.denoise.main, matfile=moire, DC=False, resultdir=resultdir), targets)
            pool.close()
            pool.join()
            end = time.time()

        else:
            start = time.time()
            for f in files:
                if f == '0-1.png':
                    continue
                target = os.path.join(datadir, f)
                # background seperation
                epll.denoise.main(target, background, True, resultdir)
                # moire seperation
                epll.denoise.main(target, moire, False, resultdir)
                remove_background.remove_background.main(target, resultdir)
            end = time.time()

        print('Elapsed time : {:.1f}s\n'.format(end-start))


    def post_process(self):
        datadir = self.post_datadir
        resultdir = self.post_resultdir

        assert os.path.exists(datadir), print('Demoire: datadir not exists')
        
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        source = os.path.join(datadir, 'source.png')
        target = os.path.join(datadir, 'target.png')
        clean = os.path.join(datadir, '0-1.png')

        start = time.time()
        post_process.post_process.main(source, target, clean, resultdir)
        end = time.time()

        info = pre_process.pre_process.snapshot_info
        level = len(info)
        img_width, img_height = info[0][1]
        filename = '0-1.png'
        filepath = os.path.join(datadir, filename)
        img_zero = cv2.imread(filepath,  cv2.IMREAD_COLOR)
        result_name = 'clean0.png'
        result_path = os.path.join(resultdir, result_name)
        cv2.imwrite(result_path, img_zero)
        for i in range(1, level):
            img_merge = np.empty(shape=(img_height, img_width, 3))
            acc_array = np.empty(shape=(img_height, img_width, 3))
            nx, ny = info[i][0]
            partial_width, partial_height = info[i][1]
            stride = info[i][2]
            num = 1
            
            for y in range(ny):
                if y != ny - 1:
                    for x in range(nx):
                        filename = str(i)+'-'+str(num)+'.png'
                        filepath = os.path.join(datadir, filename)
                        partial_image = cv2.imread(filepath,  cv2.IMREAD_COLOR)
                        temp = np.empty(shape=(partial_width, partial_height))
                        if x == nx - 1:
                            temp = img_merge[y * stride[1]: y * stride[1] + partial_height,\
                                            img_width-1-partial_width:img_width-1]
                            temp = (temp+partial_image)                         
                            img_merge[y * stride[1]: y * stride[1] + partial_height,\
                                    img_width-1-partial_width:img_width-1] = temp
                            acc_array[y * stride[1]: y * stride[1] + partial_height,\
                                    img_width-1-partial_width:img_width-1] += 1
                        else:
                            temp = img_merge[y * stride[1]: y * stride[1] + partial_height,\
                                            x * stride[0]: x * stride[0] + partial_width]
                            temp = (temp+partial_image)
                            img_merge[y * stride[1]: y * stride[1] + partial_height,\
                                    x * stride[0]: x * stride[0] + partial_width] = temp
                            acc_array[y * stride[1]: y * stride[1] + partial_height,\
                                    x * stride[0]: x * stride[0] + partial_width] += 1
                        num += 1
                else:
                    for x in range(nx):
                        filename = str(i)+'-'+str(num)+'.png'
                        filepath = os.path.join(datadir, filename)
                        partial_image = cv2.imread(filepath,  cv2.IMREAD_COLOR)
                        if x == nx - 1:
                            temp = img_merge[img_height-1-partial_height:img_height-1,\
                            img_width-1-partial_width:img_width-1]
                            temp = (temp+partial_image)
                            img_merge[img_height-1-partial_height:img_height-1,\
                                    img_width-1-partial_width:img_width-1] = temp
                            acc_array[img_height-1-partial_height:img_height-1,\
                                    img_width-1-partial_width:img_width-1] += 1
                        else:
                            temp = img_merge[img_height-1-partial_height:img_height-1,\
                            x * stride[0]: x * stride[0] + partial_width]
                            temp = (temp+partial_image)
                            img_merge[img_height-1-partial_height:img_height-1,\
                                    x * stride[0]: x * stride[0] + partial_width] = temp
                            acc_array[img_height-1-partial_height:img_height-1,\
                                    x * stride[0]: x * stride[0] + partial_width] += 1
                        num += 1

            result_name = 'clean'+str(i)+'.png'
            result_path = os.path.join(resultdir, result_name)
            cv2.imwrite(result_path, img_merge/acc_array)

        # post_process
        im = cv2.imread(source, cv2.IMREAD_COLOR)
        imReference = cv2.imread(target, cv2.IMREAD_COLOR)


if __name__ == '__main__':

    print()
    print('MopReM : Moiré Pattern Removal for Mobile, Texts/Diagrams on Single-colored Background')
    print('PIP team, GilHo Roh, Sangjun Son, Jiwon Lee, Dongseok choi')
    print('2021 fall SNU computer vision project')
    print()

    datadir = os.path.join(pardir, 'data')
    imgs = next(os.walk(datadir))[1]
    imgs.sort()

    print()
    print('total data number: {}'.format(len(imgs)))
    print()

    for imgdir in tqdm(imgs, total=len(imgs)):

        if imgdir == 'etc' or imgdir == '.ipynb_checkpoints':
            continue

        print()
        print('image directory : {}'.format(imgdir))
        print()

        resultdir = os.path.join(pardir, 'result', imgdir)

        pre_datadir = os.path.join(datadir, imgdir)
        pre_resultdir = os.path.join(resultdir, 'pre_process')

        epll_datadir = pre_resultdir
        epll_resultdir = os.path.join(resultdir, 'demoire')

        post_datadir = epll_resultdir
        post_resultdir = os.path.join(resultdir, 'post_process')

        model = MopReM.init_moprem( pre_datadir, pre_resultdir, 
                                    epll_datadir, epll_resultdir, 
                                    post_datadir, post_resultdir )

        # model.pre_process()

        model.demoire(  background='GSModel_8x8_200_2M_noDC_zeromean.mat',
                        moire='GMM_8x8_200_1500.mat',
                        pooling=True  )

        model.post_process()