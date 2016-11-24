# SegNet Model Zoo
This page lists a number of example SegNet models in the SegNet Model Zoo. NOTE: all Bayesian SegNet models can be tested as SegNet models (for example by using the webcam demo) by removing the line ```sample_weights_test: true``` on all Dropout layers, and setting batch size of 1.

### Driving Web Demo

This example model is used in the SegNet webdemo [http://mi.eng.cam.ac.uk/projects/segnet/]. It is trained to classify road scenes into 12 classes.

Model file: ```segnet_model_driving_webdemo.prototxt```

Weights can be downloaded from this link: [http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_weights_driving_webdemo.caffemodel]

### CamVid

These models have been trained for road scene understanding using the [CamVid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/).

 - Segnet Basic model file: ```segnet_basic_camvid.prototxt``` weights: [http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_basic_camvid.caffemodel]
 - Bayesian Segnet model file: ```bayesian_segnet_camvid.prototxt``` weights: [http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/bayesian_segnet_camvid.caffemodel]
 - Bayesian Segnet Basic model file: ```bayesian_segnet_basic_camvid.prototxt``` weights: [http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/bayesian_segnet_basic_camvid.caffemodel]

### SUN

These models have been trained for indoor scene understanding using the [SUN RGB-D dataset](http://rgbd.cs.princeton.edu/).

 - Segnet model file: ```segnet_sun.prototxt``` weights: [http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_sun.caffemodel]
 - Segnet model file: ```bayesian_segnet_sun.prototxt``` weights: [http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_sun.caffemodel]

The model definition file used for training can be found here ```train_segnet_sun.prototxt```

We have also trained a model for a 224x224 pixel input:

 - Segnet low resolution model file: ```segnet_sun_low_resolution.prototxt``` weights: [http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_sun_low_resolution.caffemodel]

### Pascal VOC

These models have been trained on the [Pascal VOC 2012 dataset ](http://host.robots.ox.ac.uk/pascal/VOC/).

 - Segnet model file: ```segnet_pascal.prototxt``` weights: [http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_pascal.caffemodel]
 - Bayesian Segnet model file: ```bayesian_segnet_pascal.prototxt``` weights: [http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_pascal.caffemodel]

This model is based on the Dropout enc-dec variant and is designed for a 224x224 pixel input.

### CityScapes

This model finetuned the webdemo weights using the [CityScapes dataset](https://www.cityscapes-dataset.com/) (11 class version).

 - 11 Class CityScapes Model (trained by *Timo Sämann*, Aschaffenburg University of Applied Sciences) model file: ```segnet_model_driving_webdemo.prototxt``` weights: [http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_iter_30000_timo.caffemodel]

## License

These models are released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here:
http://creativecommons.org/licenses/by-nc/4.0/

## Contact

Alex Kendall
agk34@cam.ac.uk
Cambridge University
