import os
import time
import numpy as np
from PIL import Image

import epll.denoise
import pre_process.pre_process

pardir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


class MopReM:
    def __init__(self, n):
        self.n = n


    def pre_process(self):
        datadir = os.path.join(pardir, 'data', self.n)
        assert os.path.exists(datadir), print('Pre process: datadir not exists')
        resultdir = os.path.join(pardir, 'result/pre_process', self.n)
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        source = os.path.join(datadir, 'source.png')
        target = os.path.join(datadir, 'target.png')
        
        start = time.time()
        pre_process.pre_process.main(source, target, resultdir)
        end = time.time()
        print('Pre processing time: {:.1f}s\n'.format(end-start))


    def demoire(self):
        datadir = os.path.join(pardir, 'result/pre_process', self.n)
        assert os.path.exists(datadir), print('Demoire: datadir not exists')
        files = next(os.walk(datadir))[2]
        files.sort()
        resultdir = os.path.join(pardir, 'result/demoire', self.n)
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        start = time.time()
        for f in files:
            target = os.path.join(datadir, f)
            epll.denoise.main(target, resultdir)
        end = time.time()
        print('Demoiring time: {:.1f}s\n'.format(end-start))


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
        end = time.time()
        print('Post processing time: {:.1f}s\n'.format(end-start))


if __name__ == '__main__':
    datadir = os.path.join(pardir, 'data')
    childlist = next(os.walk(datadir))[1]
    childlist.sort()
    print('Total data number: ', len(childlist))

    start = time.time()
    for dirname in childlist:
        if dirname == 'etc':
            continue
        print('\ncurrent directory : {}\n'.format(dirname))
        loop_start = time.time()
        model = MopReM(dirname)
        model.pre_process()
        model.demoire()
        model.post_process()
        loop_end = time.time()
        print('Data processing time: {}s\n'.format(loop_end-loop_start))
    end = time.time()
    print('Total time: {}s'.format(end-start))
