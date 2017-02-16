/*
 *
 *  Created on: Jan 21, 2017
 *      Author: Timo Sämann
 *
 *  The basis for the creation of this script was the classification.cpp example (caffe/examples/cpp_classification/classification.cpp)
 *
 *  This script visualize the semantic segmentation for your input image.
 *
 *  To compile this script you can use a IDE like Eclipse. To include Caffe and OpenCV in Eclipse please refer to
 *  http://tzutalin.blogspot.de/2015/05/caffe-on-ubuntu-eclipse-cc.html
 *  and http://rodrigoberriel.com/2014/10/using-opencv-3-0-0-with-eclipse/ , respectively
 *
 *
 */



#define USE_OPENCV 1
#include <caffe/caffe.hpp>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <chrono> //Just for time measurement

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;



class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file);


  void Predict(const cv::Mat& img, string LUT_file);

 private:
  void SetMean(const string& mean_file);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

  void Visualization(Blob<float>* output_layer, string LUT_file);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;

};

Classifier::Classifier(const string& model_file,
                       const string& trained_file) {


  Caffe::set_mode(Caffe::GPU);

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}


void Classifier::Predict(const cv::Mat& img, string LUT_file) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);


  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); //Just for time measurement

  net_->Forward();

  std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
  std::cout << "Processing time = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000.0 << " sec" <<std::endl; //Just for time measurement


  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];

  Visualization(output_layer, LUT_file);
}


void Classifier::Visualization(Blob<float>* output_layer, string LUT_file) {

  std::cout << "output_blob(n,c,h,w) = " << output_layer->num() << ", " << output_layer->channels() << ", "
			  << output_layer->height() << ", " << output_layer->width() << std::endl;

  cv::Mat merged_output_image = cv::Mat(output_layer->height(), output_layer->width(), CV_32F, const_cast<float *>(output_layer->cpu_data()));
  //merged_output_image = merged_output_image/255.0;

  merged_output_image.convertTo(merged_output_image, CV_8U);
  cv::cvtColor(merged_output_image.clone(), merged_output_image, CV_GRAY2BGR);
  cv::Mat label_colours = cv::imread(LUT_file,1);
  cv::Mat output_image;
  LUT(merged_output_image, label_colours, output_image);

  cv::imshow( "Display window", output_image);
  cv::waitKey(0);
}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " \ndeploy.prototxt \nnetwork.caffemodel"
              << " \nimg.jpg" << " \ncamvid12.png (for example: /SegNet-Tutorial/Scripts/camvid12.png)" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2]; //for visualization


  Classifier classifier(model_file, trained_file);

  string file = argv[3];
  string LUT_file = argv[4];

  std::cout << "---------- Semantic Segmentation for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, 1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  cv::Mat prediction;

  classifier.Predict(img, LUT_file);
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV


