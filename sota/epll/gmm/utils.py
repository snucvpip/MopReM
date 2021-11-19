import numpy as np
from PIL import Image


def load_image(paths, size=(-1, -1), n=2):
    assert len(paths) >= n, print("load_image error")
    
    datas = []
    for path in paths[:n]:
        img = Image.open(path)
        img.load()
        datas.append(np.asarray(img, dtype=np.int32))
        img.close()

    # slice image for same size
    if size[0] == -1 and size[1] == -1:
        size = datas[0].shape
    for i in range(len(datas)):
        datas[i] = datas[i][:size[0], :size[1], :]

    return datas


def flat_image(img):
    flatten = img.reshape(-1, img.shape[2]).T
    r = flatten[0, :]
    g = flatten[1, :]
    b = flatten[2, :]

    return (r, g, b)
    

def make_data(imgs):
    r_channel = []
    g_channel = []
    b_channel = []

    for img in imgs:
        r, g, b = flat_image(img)
        r_channel.append(r)
        g_channel.append(g)
        b_channel.append(b)

    return (np.asarray(r_channel), np.asarray(g_channel), np.asarray(b_channel))