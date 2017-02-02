#ifndef CAFFE_SIAMESE_ACCURACY_LAYER_HPP_
#define CAFFE_SIAMESE_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

  // class declaration 
template <typename Dtype>
class SiameseAccuracyLayer : public Layer<Dtype> {
 public:
  
  // constructor 
  explicit SiameseAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param){}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  			  const vector<Blob<Dtype>*>& top);
  
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		       const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline const char* type() const { return "SiameseAccuracy"; }

 protected:
 
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			   const vector<Blob<Dtype>*>& top);
  
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);
  
  /**  
   * For siamese accuracy layer, we do not need backward pass    
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			    const vector<bool>& propagate_down, 
			    const vector<Blob<Dtype>*>& bottom)
  {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

};

}  // namespace caffe

#endif  // CAFFE_SIAMESE_ACCURACY_LAYER_HPP_
