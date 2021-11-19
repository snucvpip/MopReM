import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn

from gmm import GMM
import utils
import os
import time

# https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f
# Christopher M. Bishop 2006, Pattern Recognition and Machine Learning
# Python for Data Science Handbook

def train(model : GMM, X, plot=False):
    fitted_values = model.fit(X)
    # print(fitted_values)
    predicted_values = model.predict(X)

    centers = np.zeros((3,2))
    for i in range(model.C):
        density = mvn(cov=model.sigma[i], mean=model.mu[i]).logpdf(X)
        centers[i, :] = X[np.argmax(density)]
    
    if plot:
        plt.figure(figsize = (10,8))
        plt.scatter(X[:, 0], X[:, 1],c=predicted_values ,s=50, cmap='viridis')

        plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6)


if __name__ == '__main__':
    start = time.time()
    paths = [os.path.join('./images', 'deer.jpeg'), 
             os.path.join('./images', 'fox.jpeg'),
             os.path.join('./images', 'deer2.jpeg')]
    imgs = utils.load_image(paths, (10,10), len(paths))
    r_data, g_data, b_data = utils.make_data(imgs)

    r_model = GMM(3, n_runs = 100)
    g_model = GMM(3, n_runs = 100)
    b_model = GMM(3, n_runs = 100)

    train(r_model, r_data)
    train(g_model, g_data)
    train(b_model, b_data)
    end = time.time()
    print("time elapsed : {:.2f}s".format(end-start))