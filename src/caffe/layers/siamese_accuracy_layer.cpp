#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/siamese_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void SiameseAccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
					     const vector<Blob<Dtype>*>& top) 
{
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  // make sure that bottom_i and bottom_j have the same dimension
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());  
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());

  // make sure that label is in correct shape
  CHECK_EQ(bottom[2]->channels(), 1); 
  CHECK_EQ(bottom[2]->height(), 1); 
  CHECK_EQ(bottom[2]->width(), 1); 
    
  // make sure number of labels matched with number of image embeddings
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());

  
 /* Blob<Dtype>::Reshape(): 
    + Change the dimensions of the blob, allocating new memory if necessary.
  */
  _diff.Reshape(bottom[0]->shape(0), bottom[0]->channels(), 1, 1);
  _diff_sq.Reshape(bottom[0]->shape(0), bottom[0]->channels(), 1, 1);
  _dist_sq.Reshape(bottom[0]->shape(0), 1, 1, 1);
  // vector of ones used to sum along channels
  _summer_vec.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    _summer_vec.mutable_cpu_data()[i] = Dtype(1);
}


template <typename Dtype>
void SiameseAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					      const vector<Blob<Dtype>*>& top) 

{
  
  int num = bottom[0]->shape(0); 
  const int channels = bottom[0]->shape(1);
  Dtype accuracy(0.0);
  int correct_examples = 0;
  int count = bottom[0]->count();
  
  /* 
     subtraction: 
     void caffe_sub<double>(const int n, const Dtype* a, const Dtype* b, Dtype* y)
   */
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // const *Dtype a_i
      bottom[1]->cpu_data(),  // const *Dtype b_i
      _diff.mutable_cpu_data());  // a_i-b_i

  // reuse margin paramters of contrastive loss 
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
     
  // calculate accuracy 
  for(int i = 0; i < num; i++){
    
    /* 
       get square distance for each example
       Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y)
     */
    _dist_sq.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
    				     _diff.cpu_data() + (i*channels), _diff.cpu_data() + (i*channels));
    
      
    // similar pair
    if(static_cast<int>(bottom[2]->cpu_data()[i])){
      
      // if _dist_sq <= margin => correct_examples += 1
      if(_dist_sq.cpu_data()[i] <= margin){
      	correct_examples += 1;
      }      
    }

    // dissimilar pair
    else{
      
      // if _dist_sq > margin => correct_examples += 1
      if(_dist_sq.cpu_data()[i] > margin){
      	correct_examples += 1;
      }     
    }
  }
   
  accuracy = static_cast<Dtype>(correct_examples) / Dtype(num);

  // update result to top vector
  top[0]->mutable_cpu_data()[0] = accuracy;
}


#ifdef CPU_ONLY
STUB_GPU(SiameseAccuracyLayer);
#endif

INSTANTIATE_CLASS(SiameseAccuracyLayer);
REGISTER_LAYER_CLASS(SiameseAccuracy);

}  // namespace caffe
