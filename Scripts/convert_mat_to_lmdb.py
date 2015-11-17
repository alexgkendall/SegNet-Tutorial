# This script can be used to convert data which will not fit into .png into a format for SegNet. It converts Matlab files into LMDB format.
# For example:
#   - data with more than 3 channels
#   - data with floating point values

caffe_root = '/scratch/PhD_Projects/SCORE_regression/CaffeScoreRegression/'
import sys
sys.path.insert(0, caffe_root + 'python')
import numpy as np
import lmdb
import caffe
import scipy.io as sio
import h5py
import random


print 'Loading Matlab data.'
f = h5py.File('/scratch/PhD_Projects/SegNet_Classification/data/NYU/NYU_train_test_rgbdn_MoosmanNormals.mat','r')
# name of your matlab variables: 
data = f.get('data/images')
labels = f.get('data/labels')

print 'Creating label dataset.'
Y = np.array(labels)
Y = np.array(Y,dtype=np.float32)
map_size = Y.nbytes*2
N = range(Y.shape[0])

#if you want to shuffle your data
#random.shuffle(N)

env = lmdb.open('labels', map_size=map_size)
for i in N:
    im_dat = caffe.io.array_to_datum(np.array(Y[i]).astype(float))
    str_id = '{:0>10d}'.format(i)
    with env.begin(write=True) as txn:
        txn.put(str_id, im_dat.SerializeToString())


print 'Creating image dataset.'
X = np.array(data)
X = np.array(X,dtype=np.float32)
map_size = X.nbytes*2
env = lmdb.open('data', map_size=map_size)
for i in N:
    im_dat = caffe.io.array_to_datum(np.array(X[i]).astype(float))
    str_id = '{:0>10d}'.format(i)
    with env.begin(write=True) as txn:
        txn.put(str_id, im_dat.SerializeToString())
