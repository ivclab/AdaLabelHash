# AdaLabelHash
official implmemtation of [ADAPTIVE LABELING FOR HASH CODE LEARNING VIA NEURAL NETWORKS](https://ieeexplore.ieee.org/document/8803011)

Created by [Huei-Fang Yang](https://github.com/hueifang), [Cheng-Hao Tu](https://github.com/andytu28), Chu-Song Chen 

## Introduction 
Learning-based hash has been widely used for large-scale similarity retrieval due to the efficient computation and condensed storage of binary representations. In this paper, we propose AdaLabelHash, a hash function learning approach via neural networks. In AdaLabelHash, class label representations are adaptable during the network training. We express the labels as hypercube vertices in a K-dimensional space, and both the network weights and class label representations are updated in the learning process. As the label representations are explored from data, semantically similar categories will be assigned with the label representations that are close to each other in terms of Hamming distance in the label space. The label representations then serve as the desired output of the hash function learning so as to yield compact and discriminating binary hash codes via the network. AdaLabelHash is simple but effective, which can jointly learn label representations and infer compact binary codes from data. It is applicable to both supervised and semi-supervised learning of hash codes. Experimental results on standard benchmarks show the effectiveness of AdaLabelHash.


## Prerequisites 
* python2.7
* keras2.3.0
* tensorflow1.13.1
* scikit-learn0.20.4

Use the following command to install all requirements: 
```bash
$ pip install -r requirements.txt
```

If you want to evaluate the image retrieval performance, you need matlab as well. 


## Usage

We present the instructions on training and testing on CIFAR10 as follows: 

### Preparing data

* You will need to download the [pretrained CNN_F](https://drive.google.com/open?id=1HJ8UdIwNt_pGricAM7LfnVeGCgMZHIY4) and place `vgg_cnn_f_rmlast.h5` in `init_weights/`. 
We convert the pretrained weights from Caffe so that we can load them using Keras. The Caffe version of weights are provided [here](https://gist.github.com/ksimonyan/a32c9063ec8e1118221a). 

* For CIFAR10, we convert the numpy formats provided in the [official website](https://www.cs.toronto.edu/~kriz/cifar.html) into the png formats. 
Download the [converted CIFAR10](https://drive.google.com/open?id=1Zy72S74AGDAX-OjLqR5NrVUmMCy653pT), extract the .zip file and place `cifar/` in `data/`. 

### Training 

### Testing 

### Evaluation (matlab required)


## Citation 
Please cite following paper if these codes help your research:
    
    @inproceedings{yang2019adaptive,
        title={Adaptive Labeling For Hash Code Learning Via Neural Networks},
        author={Yang, Huei-Fang and Tu, Cheng-Hao and Chen, Chu-Song},
        booktitle={2019 IEEE International Conference on Image Processing (ICIP)},
        pages={2244--2248},
        year={2019},
        organization={IEEE}
    }

    @inproceedings{yang2019adaptive,
        title={Adaptive Labeling for Deep Learning to Hash},
        author={Yang, Huei-Fang and Tu, Cheng-Hao and Chen, Chu-Song},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
        pages={0--0},
        year={2019}
    }


## Contact 
Please feel free to leave suggestions or comments to [Huei-Fang Yang](https://sites.google.com/site/hueifang/home), Cheng-Hao Tu(andytu28@iis.sinica.edu.tw), Chu-Song Chen(song@iis.sinica.edu.tw)
