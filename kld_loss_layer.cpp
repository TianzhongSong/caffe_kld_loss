#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/kld_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KLDLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void KLDLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
    << "Input bottoms must have the same size.";
  outer_num_ = bottom[0]->shape(0);
  inner_num_ = bottom[0]->count(1);
}

template <typename Dtype>
Dtype KLDLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, Dtype valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count < 0) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = valid_count;
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void KLDLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* prob_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype * temp = bottom[0]->mutable_cpu_diff();

  int N = bottom[0]->count();
  Dtype loss = 0;

  // Prevent underflow in log
  for(int i = 0; i < N; ++i)
    temp[i] = std::max(prob_data[i],Dtype(FLT_MIN));
  caffe_log(N,temp,temp);

  // Subtract entropy of target distribution
  for(int i = 0; i < N; ++i)
    temp[i] -= log(std::max(label[i],Dtype(FLT_MIN)));

  loss = caffe_cpu_dot(N,temp,label);
  top[0]->mutable_cpu_data()[0] = -loss / 
      get_normalizer(normalization_, Dtype(N));
}

template <typename Dtype>
void KLDLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to target distribution yet.";
  }

  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    int N = bottom[0]->count();

    caffe_sub(N,prob_data,label,bottom_diff);
    Dtype loss_weight = top[0]->cpu_diff()[0] / 
        get_normalizer(normalization_, Dtype(N));
    caffe_scal(N, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
  STUB_GPU(KLDLossLayer);
#endif

INSTANTIATE_CLASS(KLDLossLayer);
REGISTER_LAYER_CLASS(KLDLoss);

}  // namespace caffe