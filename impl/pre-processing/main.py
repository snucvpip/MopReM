from pre_processing import *
import os
import math

dirName = "data/sample"
imFilename = os.path.join(dirName, "source.jpg")
# image = cv2.imread(imFilename)

from skimage.feature import peak_local_max

'''
def GetLimitNxNy(img):
    pim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = pim.shape
    n_pixel = 10
        
    coordinates = np.array(peak_local_max(pim, min_distance=3))
    M = np.ceil(np.sqrt(coordinates.shape[0]))
    N = coordinates.shape[0]
    
    d = np.ceil(np.sqrt(width*height/N))
    
    return tuple(np.array([width, height]) // int(d * n_pixel))

dirName = "data/snipping"
imFilename = os.path.join(dirName, "source.png")

print("Reading image to align : ", imFilename);  
im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
nx, ny = GetLimitNxNy(im)

pim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
height, width = pim.shape

plt.figure(figsize=(15,15))
plt.imshow(pim[1000:1000+(height//ny), 1000:1000+(width//nx)], 'gray')
'''



def Snapshot(image, lim_scale, level=10, stride=None):
    #################### EXPLANATION #####################
    # This function makes snapshots by dividing whole input image according to scale.
    # We can get lim_scale by GetLimScale() function.
    ######################################################

    ##################### PARAMETERS #####################
    # image: phone-captured image
    # lim_scale: Maximal magnification of image in which we can recognize the moire pattern
    # level: log scale level in range of 1 ~ lim_scale
    # stride: (stride_x, stride_y)
    ######################################################

    ################### USAGE EXAMPLE ####################
    # If you put in 40 as lim_scale and 10 as level,
    # this function will make 10 snapshot sets according to incremental log scale as below,
    # [ 1.  1.50663019  2.26993453  3.41995189  5.15260277  7.76306689, 11.69607095 17.62165361 26.54931532 40.]
    ######################################################

    scale = np.exp(np.linspace(math.log(1), math.log(lim_scale), level))

    for i in range(level):
        dir_path = dirName + '/snapshot_level' + str(i)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        width = int(math.ceil(image.shape[1] / scale[i]))
        height = int(math.ceil(image.shape[0] / scale[i]))

        if stride is None:
            stride_i = (width, height)

        num_image = 1
        y_start, y_end = (0, height)

        while y_end <= image.shape[0]:
            x_start, x_end = (0, width)

            while x_end <= image.shape[1]:
                cv2.imwrite(dir_path + '/' + str(num_image) + '.jpg', image[y_start:y_end, x_start:x_end])
                x_start += stride_i[0]
                x_end += stride_i[0]
                num_image += 1

            y_start += stride_i[1]
            y_end += stride_i[1]


# if __name__ == "__main__":
#     Snapshot(40)
