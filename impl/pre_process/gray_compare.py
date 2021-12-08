from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import os
import cv2
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np

datadir = './data'
src_name = 'background.png'
tar_name = 'moire.png'

# grayscale
src = cv2.imread(os.path.join(datadir, src_name), cv2.IMREAD_GRAYSCALE)
tar = cv2.imread(os.path.join(datadir, tar_name), cv2.IMREAD_GRAYSCALE)

mse = mean_squared_error(src, tar)
psnr = peak_signal_noise_ratio(src, tar)
ssim, diff = structural_similarity(src, tar, multichannel=True, full=True)

diff = (diff * 255).astype("uint8")

fig, ax = plt.subplots(ncols=4, figsize=(15, 5))
ax[0].imshow(src, 'gray')
ax[0].set_title('Source')
ax[1].imshow(tar, 'gray')
ax[1].set_title('Target')
ax[2].imshow(np.abs(src-tar), 'gray')
ax[2].set_title(f'MSE:{round(mse,2)}, PSNR:{round(psnr,2)}, SSIM:{round(ssim,2)}')
ax[3].imshow(diff, cmap='gray')
ax[3].set_title('Difference')
plt.savefig(os.path.join(datadir, 'result_gray.png'))