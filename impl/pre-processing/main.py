from pre_processing import *
import os
import math

dirName = "data/sample"
imFilename = os.path.join(dirName, "source.jpg")
image = cv2.imread(imFilename)


def Snapshot(lim_scale, level=10, stride=None):
    # stride = (stride_x, stride_y)

    scale = np.exp(np.linspace(math.log(1), math.log(lim_scale), level))

    for i in range(level):
        dir_path = dirName + '/snapshot_level' + str(i)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        width = int(math.ceil(image.shape[1] / scale[i]))
        height = int(math.ceil(image.shape[0] / scale[i]))

        if stride is None:
            stride = (width, height)

        x_start, x_end = (0, width)
        y_start, y_end = (0, height)

        num_image = 1

        while y_end <= image.shape[0]:
            while x_end <= image.shape[1]:
                cv2.imwrite(dir_path + '/' + str(num_image) + '.jpg', image[y_start:y_end, x_start:x_end])
                x_start += stride[0]
                x_end += stride[0]
                num_image += 1

            y_start += stride[1]
            y_end += stride[1]


if __name__ == "__main__":
    Snapshot(40)
