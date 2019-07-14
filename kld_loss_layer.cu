#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/kld_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void KLD_fwd_gpu(const int N,
          const Dtype* p, const Dtype* q, Dtype* loss) {
  CUDA_KERNEL_LOOP(index, N) {
    loss[index] = log(max(p[index],Dtype(FLT_MIN)));
    loss[index] -= log(max(q[index],Dtype(FLT_MIN)));

  }
}

template <typename Dtype>
void KLDLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* prob_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  Dtype* temp = bottom[0]->mutable_gpu_diff();

  int N = bottom[0]->count();
  Dtype loss = 0;
  
  KLD_fwd_gpu<Dtype><<<CAFFE_GET_BLOCKS(N),CAFFE_CUDA_NUM_THREADS>>>(
     N,prob_data,label,temp);
  caffe_gpu_dot(N,temp,label,&loss);

  top[0]->mutable_cpu_data()[0] = -loss / 
      get_normalizer(normalization_, Dtype(N));
}

template <typename Dtype>
void KLDLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to target distribution yet.";
  }

  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = bottom[0]->gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    int N = bottom[0]->count();
    
    caffe_gpu_sub(N,prob_data,label,bottom_diff);

    Dtype loss_weight = top[0]->cpu_diff()[0] /
      get_normalizer(normalization_, Dtype(N));

    caffe_gpu_scal(N, loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(KLDLossLayer);

}  // namespace caffe