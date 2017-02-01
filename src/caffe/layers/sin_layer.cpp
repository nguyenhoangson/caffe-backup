// Sin neuron activation function layer.
// Adapted from TanH layer which was adapted from the ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/sin_layer.hpp"

namespace caffe {

/* 
We take in an immutable reference of the bottom (what our layer will be getting input from), and a mutable reference to the top (what our layer will be outputting to). We will transform the input data and store it in the output  
*/
template <typename Dtype>
void SinLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) 
{ 
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  for (int i = 0; i < count; ++i) {
    top_data[i] = sin(bottom_data[i]);
  }
}

/*
We take in an immutable reference of the bottom (what our layer had previously received input from), an immutable copy of the derivative from the layer above us, and a mutable reference to the gradient that we will output. We calculate the gradient, applying the chain rule (a multiplication with the previous calculation), and set the output gradient (bottom_diff).
*/
template <typename Dtype>
void SinLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<Dtype>*>& bottom) 
{ 
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype bottom_datum;

    for (int i = 0; i < count; ++i) {
      bottom_datum = bottom_data[i];
      bottom_diff[i] = top_diff[i] * cos(bottom_datum);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SinLayer);
#endif

INSTANTIATE_CLASS(SinLayer);
REGISTER_LAYER_CLASS(Sin);

}  // namespace caffe    
