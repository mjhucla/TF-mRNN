# TF-mRNN: a TensorFlow library for image captioning.

Created by [Junhua Mao](www.stat.ucla.edu/~junhua.mao)

## Introduction

This package is a re-implementation of the [m-RNN](http://www.stat.ucla.edu/~junhua.mao/m-RNN.html) image captioning method
using [TensorFlow](https://www.tensorflow.org/).
The training speed is optimized with buckets of different lengths of the training sentences.
It also support the *Beam Search* method to decode image features into 
sentences.

## Citing m-RNN

If you find this package useful in your research, please consider citing:

    @article{mao2014deep,
      title={Deep Captioning with Multimodal Recurrent Neural Networks (m-RNN)},
      author={Mao, Junhua and Xu, Wei and Yang, Yi and Wang, Jiang and Huang, Zhiheng and Yuille, Alan},
      journal={ICLR},
      year={2015}
    }
    
## Requirements
- [TensorFlow](https://www.tensorflow.org/) 0.8+
- python 2.7 (Need ackages of numpy, scipy, nltk. All included in [Anaconda](https://store.continuum.io/cshop/anaconda/))
- [MS COCO caption toolkit](https://github.com/tylin/coco-caption)

## Basic installation (sufficient for the demo)
1. install [MS COCO caption toolkit](https://github.com/tylin/coco-caption)

2. Suppose that toolkit is install on $PATH_COCOCap and this package is install at $PATH_mRNN_CR. Create a soft link to COCOCap as follows:
  ```Shell
  cd $PATH_mRNN_CR
  ln -sf $PATH_COCOCap ./external/coco-caption
  ```
  
3. Download necessary data for using a trained m-RNN model.
  ```Shell
  bash setup.sh
  ```
  
## Demo
This demo shows how to use a trained model to generate descriptions for an image.
Run *demo.py* or view *demo.ipynb*

The configuration of the trained model is: ./model_conf/mrnn_GRU_conf.py.

The model achieves a CIDEr of 0.890 and a BLEU-4 of 0.282 on the 1000 validation images used in the [m-RNN](http://arxiv.org/abs/1412.6632) paper.
It adopts a [transposed weight sharing](http://arxiv.org/abs/1504.06692) strategy that accelerates the training and regularizes the network.


## Training your own models on MS COCO
### Download or extract image features for images in MS COCO.
Use the following shell to download extracted image features ([Inception-v3](http://arxiv.org/abs/1512.00567) or [VGG](http://arxiv.org/abs/1409.1556)) for MS COCO.
  ```Shell
  # If you want to use inception-v3 image feature, then run:
  bash ./download_coco_inception_features.sh
  # If you want to use VGG image feature, then run:
  bash ./download_coco_vgg_features.sh
  ```

Alternatively, you can extract image features yourself, you should download images from [MS COCO](http://mscoco.org/dataset/#download) dataset first.
Please make sure that we can find the image on ./datasets/ms_coco/images/ (should have at least train2014 and val2014 folder).
After that, type:
  ```Shell
  python ./exp/ms_coco_caption/extract_image_features_all.py
  ```

### Generate dictionary.
  ```Shell
  python ./exp/ms_coco_caption/create_dictionary.py
  ```

### Train and evaluate your model.
  ```Shell
  python ./exp/ms_coco_caption/mrnn_trainer_mscoco.py
  ```
  In the training, you can see the loss of your model, but it sometimes very
  helpful to see the metrics (e.g. BLEU) of the generated sentences for all
  the checkpoints of the model.
  You can simply open another terminal:
  ```Shell
  python ./exp/ms_coco_caption/mrnn_validator_mscoco.py
  ```
  The trained model, and the evaluation results, are all shown in ./cache/models/mscoco/


## Training your models on other datasets
You should arrange the annotation of the other datasets in the same format of our MS COCO annotation format.
See ./datasets/ms_coco/mscoco_anno_files/README.md for details.


## TODO
1. Allow end-to-end finetuning of the vision network parameters.
