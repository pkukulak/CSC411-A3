import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage.viewer import ImageViewer

from util import load_data

import cPickle as pickle


def compute_gabor(image):
	images = []
	for theta in np.arange(0.1, np.pi, np.pi / 8):
		for lmda in np.arange(0.1, np.pi, np.pi/4): 

			imaginary, real = gabor_filter(image, .5,theta=theta,bandwidth=lmda)
			images.append(real)

	return images

def compute_feats(image, kernels):
	feats = []
	for k, kernel in enumerate(kernels):
		filtered = ndi.convolve(image, kernel, mode='wrap')
		feats.append(filtered.flatten())
	return np.vstack(feats).reshape(1,len(kernels)*image.size)

def gabor_features(data_in):
	#create gabor kernels
	kernels = []
	for theta in range(8):
		theta = theta / 8.0 * np.pi
		for sigma in (1, 3, 5):
			for frequency in (.10, 0.25, 1.0):
				kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
				kernels.append(kernel)

	#iterate over data points, and compute the result of applying each gabor kernel
	m,n = data_in.shape
	gabor_features = []
	for i in range(data_in.shape[0]):
		gabor_features.append(compute_feats(data_in[i].reshape(32,32), kernels))
	
	return np.vstack(gabor_features)

if __name__ == "__main__":
    data_in, data_targ, data_ids, unlabeled_in, test_in = load_data()
    print "Applying Gabor filters to input data."
    gabor_feats = gabor_features(data_in)
    print "Apply Gabor filters to test data."
    gabor_test = gabor_features(test_in)
    print "Dumping input data."
    pickle.dump(gabor_feats, open("gabor.p", "wb"))
    print "Dumping test data."
    pickle.dump(gabor_test, open("gabor_test.p", "wb"))
