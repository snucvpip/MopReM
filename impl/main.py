import os
import time
from tqdm import tqdm
from glob import glob

import epll.denoise
import pre_process.pre_process
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
        
        start = time.time()
        pre_process.pre_process.main(source, target, resultdir)
        end = time.time()

        print('Elapsed time : {:.1f}s\n'.format(end-start))


    def demoire(self):
        datadir = self.epll_datadir
        resultdir = self.epll_resultdir

        assert os.path.exists(datadir), print('Demoire: datadir not exists')

        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        files = next(os.walk(datadir))[2]
        files.sort()
        
        start = time.time()
        for f in files:
            # target = os.path.join(datadir, f)
            target = os.path.join(datadir, '2-23.png')
            epll.denoise.main(target, resultdir)
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

        print('Elapsed time : {:.1f}s\n'.format(end-start))


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

        model.pre_process()
        model.demoire()
        model.post_process()