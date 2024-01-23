# :camel: WENO
Official PyTorch implementation of our NeurIPS 2022 paper: **[Bi-directional Weakly Supervised Knowledge Distillation for Whole Slide Image Classification](https://arxiv.org/abs/2210.03664)**. We propose an end-to-end weakly supervised knowledge distillation framework (**WENO**) for WSI classification, which integrates a bag classifier and an instance classifier in a knowledge distillation framework to mutually improve the performance of both classifiers. WENO is a plug-and-play framework that can be easily applied to any existing attention-based bag classification methods.

<p align="center">
  <img src="https://github.com/miccaiif/WENO/blob/main/figure3.jpg" width="640">
</p>

### Frequently Asked Questions.

* Regarding the preprocessing

  For specific preprocessing, as the settings of different MIL experiments vary (such as patch size, scale, etc.), patching needs to be conducted according to your own experimental settings. The [DSMIL](https://github.com/binli123/dsmil-wsi) paper provides a good example for reference (and is also referenced in this article). As uploading all these extracted feats files would require a lot of time and space, we have open-sourced the main and key code models. The training details in the paper and main codes can support the reproduction of this work. Thank you again for your attention! You are welcome to contact and cite us! Thank you!

### Citation
If this work is helpful to you, please cite it as:
```
@article{qu2022bi,
  title={Bi-directional weakly supervised knowledge distillation for whole slide image classification},
  author={Qu, Linhao and Wang, Manning and Song, Zhijian and others},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={15368--15381},
  year={2022}
}
```

### Contact Information
If you have any question, please email to me [lhqu20@fudan.edu.cn](lhqu20@fudan.edu.cn).
