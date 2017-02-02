#include <functional>
#include <utility>
#include <vector>

#include <iostream>
#include "caffe/layers/siamese_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void SiameseAccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
					     const vector<Blob<Dtype>*>& top) 
{
  // make sure that bottom_i and bottom_j have the same dimension

  // make sure that the number of labels are the same with the number of pairs

  
}

template<typename Dtype>
void SiameseAccuracyLayer<Dtype>:: Reshape(const vector<Blob<Dtype>*>& bottom, 
					   const vector<Blob<Dtype>*>& top)
{
  std::cout << "Tesing Reshape" << std::endl; 
} 


template <typename Dtype>
void SiameseAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					      const vector<Blob<Dtype>*>& top) 

{
  // get distance of each embedding 
  

  // check if it exceeds margin 
  

  // update result to top vector
}

INSTANTIATE_CLASS(SiameseAccuracyLayer);
REGISTER_LAYER_CLASS(SiameseAccuracy);

}  // namespace caffe
