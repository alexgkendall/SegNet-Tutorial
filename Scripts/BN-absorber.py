#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
After SegNet has been trained, run compute_bn_statistics.py script and then BN-absorber.py.

For inference batch normalization layer can be merged into convolutional kernels, to
speed up the network. Both layers applies a linear transformation. For that reason
the batch normalization layer can be absorbed in the previous convolutional layer
by modifying its weights and biases. That is exactly what the script does.
"""
import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
caffe_root = '/SegNet/caffe-segnet-cudnn5/'  # Change this to the absolute directory to SegNet Caffe
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


__author__ = 'Timo Sämann'
__university__ = 'Aschaffenburg University of Applied Sciences'
__email__ = 'Timo.Saemann@gmx.de'
__data__ = '6th May, 2017'


def copy_double(data):
    return np.array(data, copy=True, dtype=np.double)


def bn_absorber_weights(model, weights):

    # load the prototxt file as a protobuf message
    with open(model) as f:
        str2 = f.read()
    msg = caffe_pb2.NetParameter()
    text_format.Merge(str2, msg)

    # load net
    net = caffe.Net(model, weights, caffe.TEST)

    # iterate over all layers of the network
    for i, layer in enumerate(msg.layer):

        # check if conv layer exist right before bn layer, otherwise merging is not possible and skip
        if not layer.type == 'BN':
            continue
        if not msg.layer[i-1].type == 'Convolution':
            continue

        # get the name of the bn and conv layer
        bn_layer = msg.layer[i].name
        conv_layer = msg.layer[i-1].name

        # get some necessary sizes
        kernel_size = 1
        shape_of_kernel_blob = net.params[conv_layer][0].data.shape
        number_of_feature_maps = list(shape_of_kernel_blob[0:1])
        shape_of_kernel_blob = list(shape_of_kernel_blob[1:4])
        for x in shape_of_kernel_blob:
            kernel_size *= x

        weight = copy_double(net.params[conv_layer][0].data)
        bias = copy_double(net.params[conv_layer][1].data)

        # receive new_gamma and new_beta which was already calculated by the compute_bn_statistics.py script
        new_gamma = net.params[bn_layer][0].data[...]
        new_beta = net.params[bn_layer][1].data[...]

        # manipulate the weights and biases over all feature maps:
        # weight_new = weight * gamma_new
        # bias_new = bias * gamma_new + beta_new
        # for more information see https://github.com/alexgkendall/caffe-segnet/issues/109
        for j in xrange(number_of_feature_maps[0]):

            net.params[conv_layer][0].data[j] = weight[j] * np.repeat(new_gamma.item(j), kernel_size).reshape(
                net.params[conv_layer][0].data[j].shape)
            net.params[conv_layer][1].data[j] = bias[j] * new_gamma.item(j) + new_beta.item(j)

        # set the no longer needed bn params to zero
        net.params[bn_layer][0].data[:] = 0
        net.params[bn_layer][1].data[:] = 0

    return net


def bn_absorber_prototxt(model):

    # load the prototxt file as a protobuf message
    with open(model) as k:
        str1 = k.read()
    msg1 = caffe_pb2.NetParameter()
    text_format.Merge(str1, msg1)

    # search for bn layer and remove them
    for l in msg1.layer:
        if l.type == "BN":
            msg1.layer.remove(l)

    return msg1


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file which you want to use for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file in which the batch normalization '
                                                                   'and convolutional layer should be merged')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='output directory in which the modified model and weights should be stored')
    parser.add_argument('--gpu', type=str, default='0', help='0: gpu mode active, else gpu mode inactive')
    return parser


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()
    if args.gpu == 0:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # check if output directory exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    network = bn_absorber_weights(args.model, args.weights)  # merge bn layer into conv kernel
    msg_proto = bn_absorber_prototxt(args.model)  # remove bn layer from prototxt file

    # save prototxt for inference
    print "Saving inference prototxt file..."
    path = os.path.join(args.out_dir, "bn_conv_merged_model.prototxt")
    with open(path, 'w') as m:
        m.write(text_format.MessageToString(msg_proto))

    # save weights
    print "Saving new weights..."
    network.save(os.path.join(args.out_dir, "bn_conv_merged_weights.caffemodel"))
    print "Done!"
