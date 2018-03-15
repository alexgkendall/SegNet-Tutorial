# SegNet and Bayesian SegNet Tutorial

This repository contains all the files for you to complete the 'Getting Started with SegNet' and the 'Bayesian SegNet' tutorials here:
http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html

Please see this link for detailed instructions.

## Caffe-SegNet

SegNet requires a modified version of Caffe to run. Please download and compile caffe-segnet to use these models:
https://github.com/alexgkendall/caffe-segnet

This version supports cudnn v2 acceleration. @TimoSaemann has a branch supporting a more recent version of Caffe (Dec 2016) with cudnn v5.1:
https://github.com/TimoSaemann/caffe-segnet-cudnn5

## Getting Started with Live Demo

If you would just like to try out an example model, then you can find the model used in the [SegNet webdemo](http://mi.eng.cam.ac.uk/projects/segnet/) in the folder ```Example_Models/```. You will need to download the weights separately using the link in the [SegNet Model Zoo](https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/segnet_model_zoo.md).

First open ```Scripts/webcam_demo.py``` and edit line 14 to match the path to your installation of SegNet. You will also need a webcam, or alternatively edit line 39 to input a video file instead. To run the demo use the command:

```python Scripts/webcam_demo.py --model Example_Models/segnet_model_driving_webdemo.prototxt --weights /Example_Models/segnet_weights_driving_webdemo.caffemodel --colours /Scripts/camvid12.png```

## Getting Started with Docker

Use docker to compile caffe and run the examples. In order to run caffe on the gpu using docker, please install nvidia-docker (see https://github.com/NVIDIA/nvidia-docker or using ansbile: https://galaxy.ansible.com/ryanolson/nvidia-docker/)

to run caffe on the CPU:
```
docker build -t bvlc/caffe:cpu ./cpu 
# check if working
docker run -ti bvlc/caffe:cpu caffe --version
# get a bash in container to run examples
docker run -ti --volume=$(pwd):/SegNet -u $(id -u):$(id -g) bvlc/caffe:cpu bash
```

to run caffe on the GPU:
```
docker build -t bvlc/caffe:gpu ./gpu
# check if working
docker run -ti bvlc/caffe:gpu caffe device_query -gpu 0
# get a bash in container to run examples
docker run -ti --volume=$(pwd):/SegNet -u $(id -u):$(id -g) bvlc/caffe:gpu bash
```

## Example Models

A number of example models for indoor and outdoor road scene understanding can be found in the [SegNet Model Zoo](https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/segnet_model_zoo.md).

## Publications

For more information about the SegNet architecture:

http://arxiv.org/abs/1511.02680
Alex Kendall, Vijay Badrinarayanan and Roberto Cipolla "Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding." arXiv preprint arXiv:1511.02680, 2015.

http://arxiv.org/abs/1511.00561
Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." PAMI, 2017. 

## License

This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here:
http://creativecommons.org/licenses/by-nc/4.0/


## Contact

Alex Kendall

agk34@cam.ac.uk

Cambridge University

