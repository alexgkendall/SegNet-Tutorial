# SegNet Tutorial

This repository contains all the files for you to complete the 'Getting Started with SegNet' tutorial here:
http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html

Please see this link for detailed instructions.

## Getting Started with Live Demo

If you would just like to try out an example model, then you can find the model used in the [SegNet webdemo](http://mi.eng.cam.ac.uk/projects/segnet/) in the folder ```Example_Models/driving_web_demo``` (you will need [git lfs](https://git-lfs.github.com/) to download the weights).

First open ```Scripts/webcam_demo.py``` and edit line 14 to match the path to your installation of SegNet. You will also need a webcam, or alternatively edit line 39 to input a video file instead. To run the demo use the command:

```python Scripts/webcam_demo.py --model Example_Models/driving_web_demo/driving.prototxt --weights /Example_Models/driving_web_demo/driving.caffemodel --colours /Scripts/camvid12.png```

## Publications

For more information about the SegNet architecture:

http://arxiv.org/abs/1511.02680
Alex Kendall, Vijay Badrinarayanan and Roberto Cipolla "Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding." arXiv preprint arXiv:1511.02680, 2015.

http://arxiv.org/abs/1511.00561
Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." arXiv preprint arXiv:1511.00561, 2015. 

## License

This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here:
http://creativecommons.org/licenses/by-nc/4.0/


## Contact

Alex Kendall
agk34@cam.ac.uk
Cambridge University

