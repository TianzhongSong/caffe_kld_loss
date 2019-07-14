# KL Divergence Loss Layer for Caffe

KL Divergence Loss layer for matching prob distribution

## Usage

```
cp *.cpp *.cu $CAFFE_ROOT/src/caffe/layers/
cp *.hpp $CAFFE_ROOT/include/caffe/layers/

then, make caffe

KLD Loss layer in prototxt:

layer {
  name: "kld_loss"
  type: "KLDLoss"
  bottom: "source_prob"
  bottom: "target_prob"
  top: "kld_loss"
  include { phase: TRAIN }
  propagate_down: true
  propagate_down: false
  loss_weight: 1
}
```

### TODO

```
complie: ok
result: not tested
```

### Reference

https://github.com/ayanc/mdepth/tree/master/training/layers

https://github.com/wentianli/knowledge_distillation_caffe