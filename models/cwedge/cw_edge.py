import cv2
from PIL import Image
import numpy as np

def CannyEdgeDetection(image, filename):
    edges = cv2.Canny(image, threshold1=100, threshold2=200)
    cv2.imwrite(filename + '.png', edges)

def seperate_image_space(Irgb):
    Ir, Ig, Ib = cv2.split(Irgb)
    cv2.imwrite('original_red.png', Ir)
    cv2.imwrite('original_green.png', Ig)
    cv2.imwrite('original_blue.png', Ib)

def main():
    Irgb = cv2.imread('1.png')
    Ir, Ig, Ib = cv2.split(Irgb)
    Igs = np.array(Image.open('1.png').convert('L'))
    CannyEdgeDetection(Ir, 'red')
    CannyEdgeDetection(Ig, 'green')
    CannyEdgeDetection(Ib, 'blue')

    seperate_image_space(Irgb)

if __name__ == '__main__':
    main()

