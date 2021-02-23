# Predicting gastric cancer outcome from resected lymph node histopathology images using deep learning
[Link to paper](https://doi.org/10.1038/s41467-021-21674-7)
### Abstract
N-staging is a determining factor for prognostic assessment and decision-making for stage-based cancer therapeutic strategies. Visual inspection of whole-slides of intact lymph nodes is currently the main method used by pathologists to calculate the number of metastatic lymph nodes (MLNs). Moreover, even at the same N stage, the outcome of patients varies dramatically. Here, we propose a deep-learning framework for analyzing lymph node whole-slide images (WSIs) to identify lymph nodes and tumor regions, and then to uncover tumor-area-to-MLN-area ratio (T/MLN). After training, our model’s tumor detection performance was comparable to that of experienced pathologists and achieved similar performance on two independent gastric cancer validation cohorts. Further, we demonstrate that T/MLN is an interpretable independent prognostic factor. These !ndings indicate that deep-learning models could assist not only pathologists in detecting lymph nodes with metastases but also oncologists in exploring new prognostic factors, especially those that are dif!cult to calculate manually.

### System requirements

#### Hardware Requirements

```
At least NVIDIA GTX 2080Ti
```

#### OS Requirements

This package is supported for Linux. The package has been tested on the following systems:
```
Linux: Ubuntu 16.04
```

#### Software Prerequisites

```
Python 3.7+
Numpy 1.17.2
Scipy 1.3.0
Pytorch 1.3.0+/CUDA 10.1
torchvision 0.4.1+
Pillow 6.0.0
opencv-python 4.1.0.25
openslide-python 1.1.1
Scikit-learn 0.21
```

### Installation guide

It is recommended to install the environment in the Ubuntu 16.04 system.

* First install Anconda3.

* Then install CUDA 10.x and cudnn.

* Finall intall these dependent python software library.

The installation is estimated to take 1 hour, depending on the network environment.

### Demo

#### Train Segmentation model

##### train model

```
python ./segmentation/bin/train.py 
```
##### test model
```
python ./segmentation/bin/test.py 
```
#### Train Classification model
##### train model
```
python ./classification/bin/train.py 
```
##### test model
```
python ./classification/bin/test.py 
```
#### Get T/MLN

```
python ./test/bin/get_T_MLN.py --wsi_path './tiff/test_patients/'
```

where --wsi_path is the path to all the WSI tiff of the patient you are interested.

### References
Appreciate the great work from the following repositories:

- [baidu-research/NCRF](https://github.com/baidu-research/NCRF)
- [vacancy/Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
- [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- [miguelvr/dropblock](https://github.com/miguelvr/dropblock/tree/master/dropblock)

### License

This project is covered under the **Apache 2.0 License**.
