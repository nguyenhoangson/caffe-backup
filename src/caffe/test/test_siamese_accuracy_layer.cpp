#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/siamese_accuracy_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

  // set up data to test against the class
template <typename TypeParam>
class SiameseAccuracyLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  
  // constructor
  SiameseAccuracyLayerTest()
      : blob_bottom_data_i_(new Blob<Dtype>(100, 2, 1, 1)),
        blob_bottom_data_j_(new Blob<Dtype>(100, 2, 1, 1)),
        blob_bottom_y_(new Blob<Dtype>(100, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);  // distances~=1.0 to test both sides of margin
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_i_);
    blob_bottom_vec_.push_back(blob_bottom_data_i_);
    filler.Fill(this->blob_bottom_data_j_);
    blob_bottom_vec_.push_back(blob_bottom_data_j_);

    // initialize label
    for (int i = 0; i < blob_bottom_y_->count(); ++i) {
      blob_bottom_y_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;  // 0 or 1
    }

    // create bottom and top vectors to pass to SiameseAccuracy
    blob_bottom_vec_.push_back(blob_bottom_y_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  
  // destructor
  virtual ~SiameseAccuracyLayerTest() {
    delete blob_bottom_data_i_;
    delete blob_bottom_data_j_;
    delete blob_bottom_y_;
    delete blob_top_loss_;
  }
  
  // represenation or data member 
  Blob<Dtype>* const blob_bottom_data_i_;
  Blob<Dtype>* const blob_bottom_data_j_;
  Blob<Dtype>* const blob_bottom_y_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SiameseAccuracyLayerTest, TestDtypesAndDevices);


TYPED_TEST(SiameseAccuracyLayerTest, TestForwardCPU) {
  
  // initialize class to be tested 
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SiameseAccuracyLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  // understand the behavior of code
  std::cout << "Size: " << this->blob_bottom_vec_.size() << std::endl;
  std::cout << "Height: " << this->blob_bottom_vec_[0]->shape(2);
  EXPECT_EQ(1, 1);
}

}  // namespace caffe
