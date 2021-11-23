This is the learning code and resulting models for the results presented in “Gaussian Mixtures, Natural Images and Dead Leaves” by Daniel Zoran and Yair Weiss, NIPS 2012.

The file “LearnGMMFromImages.m” shows how to learn a simple GMM from random patches extracted from images. Use this as a starting point.

The two MAT files are the 8x8 and 16x16 GMM models we used in the paper for log likelihood comparison (use the function GMMLogL.m to calculate the likelihood of a set of patches).

Included also are the energy functions for the Karklin and Lewicki model to be used with the HAIS code available on GIT.

For any comments or questions, feel free to email daniez@cs.huji.ac.il

All rights reserved - Daniel Zoran and Yair Weiss, 2012