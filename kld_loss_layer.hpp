/* KL Divergence Loss Layer for Caffe */

#ifndef CAFFE_KLD_LOSS_LAYER_HPP_
#define CAFFE_KLD_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
class KLDLossLayer : public LossLayer<Dtype> {
 public:
  explicit KLDLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "KLDLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinBottomBlobs() const { return 2; }
	

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, Dtype valid_count);

  LossParameter_NormalizationMode normalization_;

  int outer_num_, inner_num_;
};

}  // namespace caffe

#endif  // CAFFE_KLD_LOSS_LAYER_HPP_