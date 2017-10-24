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
    X = np.array(X)
    final = []
    for i in range(len(X)):
        f1 = X[i]
        f2 = f1.transpose()
        f3 = f1.dot(f2)
        f3.reshape(1,f3.shape[0]*f3.shape[1])
        final.append(f3)
    final = np.array(final)
    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(final,Y)
    with open('cnn_cnn.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)