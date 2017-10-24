import sys
#sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cv2
import tensorflow as tf
import sklearn 
import scipy
import glob
import numpy as np
import cPickle
#from lxml import etree
from PIL import Image
from keras.applications.vgg16 import VGG16
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import pdb

from sklearn.datasets import make_classification
from sklearn.mixture import GMM


#img_path = '/home/sukhad/torch-feature-extract/images/1_4.jpg'
def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]
    print N

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covars_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))

if __name__ == '__main__':
	model = VGG16(weights='imagenet', include_top=False)
	directory = '/root/Downloads/dataset/negative/*.jpg'
	filenames = glob.glob(directory)
	X = []
	Y = []
	count = 0
	for file in filenames:
		img = image.load_img(file)
		x1 = image.img_to_array(img)
		x1 = np.expand_dims(x1, axis=0)
		x1 = preprocess_input(x1)
		features = model.predict(x1)
		print features.shape
		features = features.reshape(1,512)
		features = features.tolist()
		X.append(features[0])
		count = count + 1
		Y.append(0)
		if(count == 20000):
			break
	directory = '/root/Downloads/dataset/positive/*.jpg'
	filenames = glob.glob(directory)
	for file in filenames:
		img = image.load_img(file)
		x1 = image.img_to_array(img)
		x1 = np.expand_dims(x1, axis=0)
		x1 = preprocess_input(x1)
		features = model.predict(x1)
		features = features.reshape(1,512)
		features = features.tolist()
		X.append(features[0])
		Y.append(1)
		count = count + 1
		if(count==40000):
			break
    N = 512
    K=len(X)
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 15)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    bgmm = best_gmm
	with open('gmm_model_fv.pkl','wb') as ffid:
		cPickle.dump(bgmm, ffid)
	fvs=[]
	for i in range(K):
		fv = fisher_vector(X[i], gmm)
		print fv.shape
		fvs.append(fv)
    fvs = np.array(fv)
	print fvs.shape
	clf = RandomForestClassifier(max_depth=5, random_state=0)
	clf.fit(fvs,Y)
	with open('fv_cnn.pkl', 'wb') as fid:
	    cPickle.dump(clf, fid)