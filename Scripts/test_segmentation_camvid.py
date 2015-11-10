import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
from sklearn.preprocessing import normalize
caffe_root = '/SegNet/caffe-segnet/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)


for i in range(0, args.iter):

	net.forward()

	image = net.blobs['data'].data
	label = net.blobs['label'].data
	predicted = net.blobs['prob'].data
	image = np.squeeze(image[0,:,:,:])
	output = np.squeeze(predicted[0,:,:,:])
	ind = np.argmax(output, axis=0)

	r = ind.copy()
	g = ind.copy()
	b = ind.copy()
	r_gt = label.copy()
	g_gt = label.copy()
	b_gt = label.copy()

	Sky = [128,128,128]
	Building = [128,0,0]
	Pole = [192,192,128]
	Road_marking = [255,69,0]
	Road = [128,64,128]
	Pavement = [60,40,222]
	Tree = [128,128,0]
	SignSymbol = [192,128,128]
	Fence = [64,64,128]
	Car = [64,0,128]
	Pedestrian = [64,64,0]
	Bicyclist = [0,128,192]
	Unlabelled = [0,0,0]

	label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
	for l in range(0,11):
		r[ind==l] = label_colours[l,0]
		g[ind==l] = label_colours[l,1]
		b[ind==l] = label_colours[l,2]
		r_gt[label==l] = label_colours[l,0]
		g_gt[label==l] = label_colours[l,1]
		b_gt[label==l] = label_colours[l,2]

	rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
	rgb[:,:,0] = r/255.0
	rgb[:,:,1] = g/255.0
	rgb[:,:,2] = b/255.0
	rgb_gt = np.zeros((ind.shape[0], ind.shape[1], 3))
	rgb_gt[:,:,0] = r_gt/255.0
	rgb_gt[:,:,1] = g_gt/255.0
	rgb_gt[:,:,2] = b_gt/255.0

	image = image/255.0

	image = np.transpose(image, (1,2,0))
	output = np.transpose(output, (1,2,0))
	image = image[:,:,(2,1,0)]


	#scipy.misc.toimage(rgb, cmin=0.0, cmax=255).save(IMAGE_FILE+'_segnet.png')

	plt.figure()
	plt.imshow(image,vmin=0, vmax=1)
	plt.figure()
	plt.imshow(rgb_gt,vmin=0, vmax=1)
	plt.figure()
	plt.imshow(rgb,vmin=0, vmax=1)
	plt.show()


print 'Success!'

