#include <algorithm>
#include <vector>

#include "caffe/layers/siamese_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SiameseAccuracyLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  Dtype accuracy(0.0);
  int correct_examples = 0;
  const int count = bottom[0]->count();
  	
  // substraction
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      _diff.mutable_gpu_data());  // a_i-b_i
  
  caffe_gpu_powx(
      count,
      _diff.mutable_gpu_data(),  // a_i-b_i
      Dtype(2),
      _diff_sq.mutable_gpu_data());  // (a_i-b_i)^2
  
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      _diff_sq.gpu_data(),  // (a_i-b_i)^2
      _summer_vec.gpu_data(),
      Dtype(0.0),
      _dist_sq.mutable_gpu_data());  // \Sum (a_i-b_i)^2
  
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  
  for (int i = 0; i < bottom[0]->num(); ++i) {
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      // if _dist_sq <= margin => correct_examples += 1
      if(_dist_sq.cpu_data()[i] <= margin){
      	correct_examples += 1;
      }
   
    } 
    else {  // dissimilar pairs
      // if _dist_sq > margin => correct_examples += 1
      if(_dist_sq.cpu_data()[i] > margin){
      	correct_examples += 1;
      }     
    }
  }

  accuracy = static_cast<Dtype>(correct_examples) / Dtype(bottom[0]->num());

  // update result to top vector
  top[0]->mutable_cpu_data()[0] = accuracy;
}

INSTANTIATE_LAYER_GPU_FUNCS(SiameseAccuracyLayer);

}  // namespace caffe
