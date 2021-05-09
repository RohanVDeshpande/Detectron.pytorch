# Generalized Intersection over Union - PyTorch Faster/Mask R-CNN

Faster/Mask R-CNN with GIoU loss implemented in PyTorch

<div align="center">

<img src="demo/33823288584_1d21cf0a26_k-pydetectron-R101-FPN.jpg" width="700px"/>

<p> Example output of <b>e2e_mask_rcnn-R-101-FPN_2x</b> using Detectron pretrained weight.</p>

<img src="demo/33823288584_1d21cf0a26_k-detectron-R101-FPN.jpg" width="700px"/>

<p>Corresponding example output from Detectron. </p>

<img src="demo/img1_keypoints-pydetectron-R50-FPN.jpg" width="700px"/>

<p>Example output of <b>e2e_keypoint_rcnn-R-50-FPN_s1x</b> using Detectron pretrained weight.</p>

</div>

**This code follows the implementation architecture of Detectron.** Only part of the functionality is supported. Check [this section](#supported-network-modules) for more information. This code now supports **PyTorch 1.0** and **TorchVision 0.3**.

With this code, you can...

1. **Train your model from scratch.**
2. **Inference using the pretrained weight file (*.pkl) from Detectron.**

This repository is originally built on [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch). However, after many modifications, the structure changes a lot and it's now more similar to [Detectron](https://github.com/facebookresearch/Detectron). I deliberately make everything similar or identical to Detectron's implementation, so as to reproduce the result directly from official pretrained weight files.

This implementation has the following features:

- **It is pure Pytorch code**. Of course, there are some CUDA code.

- **It supports multi-image batch training**.

- **It supports multiple GPUs training**.

- **It supports two pooling methods**. Notice that only **roi align** is revised to match the implementation in Caffe2. So, use it.

- **It is memory efficient**. For data batching, there are two techiniques available to reduce memory usage: 1) *Aspect grouping*: group images with similar aspect ratio in a batch 2) *Aspect cropping*: crop images that are too long. Aspect grouping is implemented in Detectron, so it's used for default. Aspect cropping is the idea from [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), and it's not used for default.

  Besides of that, I implement a customized `nn.DataParallel ` module which enables different batch blob size on different gpus. Check [My nn.DataParallel](#my-nndataparallel) section for more details about this.

## News

- (2018/05/25) Support ResNeXt backbones.
- (2018/05/22) Add group normalization baselines.
- (2018/05/15) PyTorch0.4 is supported now !
- (2019/08/28) Support PASCAL VOC and Custom Dataset
- (2019/01/17) **PyTorch 1.0 Supported now!**
- (2019/05/30) Code rebased on **TorchVision 0.3**. Compilation is now optional!

## Getting Started
Clone the repo:

```
git clone https://github.com/roytseng-tw/mask-rcnn.pytorch.git
```

### Requirements

Tested under python3.

- python packages
  - pytorch>=1.0.0
  - torchvision>=0.3.0
  - cython>=0.29.2
  - matplotlib
  - numpy
  - scipy
  - opencv
  - pyyaml
  - packaging
  - [pycocotools](https://github.com/cocodataset/cocoapi)  — for COCO dataset, also available from pip.
  - tensorboardX  — for logging the losses in Tensorboard
- An NVIDIA GPU and CUDA 8.0 or higher. Some operations only have gpu implementation.
- **NOTICE**: different versions of Pytorch package have different memory usages.

### Compilation [Optional]

Compile the CUDA code:

```
cd lib  # please change to this directory
sh make.sh
```

It will compile all the modules you need, including NMS. (Actually gpu nms is never used ...)

Note that, If you use `CUDA_VISIBLE_DEVICES` to set gpus, **make sure at least one gpu is visible when compile the code.**

### Data Preparation

Create a data folder under the repo,

```
cd {repo_root}
mkdir data
```

- **COCO**:
  Download the coco images and annotations from [coco website](http://cocodataset.org/#download).

  And make sure to put the files as the following structure:
  ```
  coco
  ├── annotations
  |   ├── instances_minival2014.json
  │   ├── instances_train2014.json
  │   ├── instances_train2017.json
  │   ├── instances_val2014.json
  │   ├── instances_val2017.json
  │   ├── instances_valminusminival2014.json
  │   ├── ...
  |
  └── images
      ├── train2014
      ├── train2017
      ├── val2014
      ├── val2017
      ├── ...
  ```
  Download coco mini annotations from [here](https://s3-us-west-2.amazonaws.com/detectron/coco/coco_annotations_minival.tgz).
  Please note that minival is exactly equivalent to the recently defined 2017 val set. Similarly, the union of valminusminival and the 2014 train is exactly equivalent to the 2017 train set.

   Feel free to put the dataset at any place you want, and then soft link the dataset under the `data/` folder:

   ```
   ln -s path/to/coco data/coco
   ```
   
- **PASCAL VOC 2007 + 12**
  Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, creat softlinks in the `data/VOC<year>` folder as folows,
  ```
  VOCdevkitPATH=/path/to/voc_devkit
  mkdir -p $DETECTRON/detectron/datasets/data/VOC<year>
  ln -s /${VOCdevkitPATH}/VOC<year>/JPEGImages $DETECTRON.PYTORCH/data/VOC<year>/JPEGImages
  ln -s /${VOCdevkitPATH}/VOC<year>/json_annotations $DETECTRON.PYTORCH/data/VOC<year>/annotations
  ln -s /${VOCdevkitPATH} $DETECTRON.PYTORCH/data/VOC<year>/VOCdevkit<year>
  ```
  The directory structure of `JPEGImages` and `annotations` should be as follows,
  ```
  VOC<year>
  ├── annotations
  |   ├── train.json
  │   ├── trainval.json
  │   ├── test.json
  │   ├── ...
  |
  └── JPEGImages
      ├── <im-1-name>.jpg
      ├── ...
      ├── <im-N-name>.jpg
  ```
  **NOTE:** The `annotations` folder requires you to have PASCAL VOC annotations in COCO json format, which is available for download [here](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip). You can also convert the XML annotatinos files to JSON by running the following script,
  ```
  python tools/pascal_voc_xml2coco_json_converter.py $VOCdevkitPATH $year
  ```
  (In order to succesfully run the script above, you need to update the full path to the respective folders in the script).
  
- **Custom Dataset**
  Similar to above, create a directory named `CustomDataset` in the `data` folder and add symlinks to the `annotations` directory and `JPEGImages` as shown for Pascal Voc dataset. You also need to link the custom dataset devkit to `CustomDataDevkit`.

Recommend to put the images on a SSD for possible better training performance

### Pretrained Model

I use ImageNet pretrained weights from Caffe for the backbone networks.

- [ResNet50](https://drive.google.com/open?id=1wHSvusQ1CiEMc5Nx5R8adqoHQjIDWXl1), [ResNet101](https://drive.google.com/open?id=1x2fTMqLrn63EMW0VuK4GEa2eQKzvJ_7l), [ResNet152](https://drive.google.com/open?id=1NSCycOb7pU0KzluH326zmyMFUU55JslF)
- [VGG16](https://drive.google.com/open?id=19UphT53C0Ua9JAtICnw84PPTa3sZZ_9k)  (vgg backbone is not implemented yet)

Download them and put them into the `{repo_root}/data/pretrained_model`.

You can the following command to download them all:

- extra required packages: `argparse_color_formater`, `colorama`, `requests`

If you use this work, please consider citing:
```
@article{Rezatofighi_2018_CVPR,
  author    = {Rezatofighi, Hamid and Tsoi, Nathan and Gwak, JunYoung and Sadeghian, Amir and Reid, Ian and Savarese, Silvio},
  title     = {Generalized Intersection over Union},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2019},
}
```

## Modifications in this repository

This repository is a fork of [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch), with an implementation of GIoU and IoU loss while keeping the code as close to the original as possible. It is also possible to train the network with SmoothL1 loss as in the original code. See the options below.

### Losses

The type of bounding box loss can be configured in the configuration file as following. `MODEL.LOSS_TYPE` configures the final bounding box refinement loss. `MODEL.RPN_LOSS_TYPE` determines the type of the RPN bounding box loss. The valid options are currently: `[iou|giou|sl1]`.

```
MODEL:
  LOSS_TYPE: 'iou'
  RPN_LOSS_TYPE: 'iou'
```

Please take a look at `compute_iou` function of [lib/utils/net.py](lib/utils/net.py) for our GIoU and IoU loss implementation in PyTorch.

### Normalizers

We also implement a normalizer of bounding box refinement losses. This can be specified with the `MODEL.LOSS_BBOX_WEIGHT` and `MODEL.RPN_LOSS_BBOX_WEIGHT` parameters in the configuration file. The default value is `1.0`. We use `MODEL.LOSS_BBOX_WEIGHT` of `10.` for IoU and GIoU experiments in the paper.

```
MODEL:
  LOSS_BBOX_WEIGHT: 10.
  RPN_LOSS_BBOX_WEIGHT: 1.
```

### Network Configurations

We add sample configuration files used for our experiment in `config/baselines`. Our experiments in the paper are based on `e2e_faster_rcnn_R-50-FPN_1x.yaml` and `e2e_mask_rcnn_R-50-FPN_1x.yaml` as following:

```
e2e_faster_rcnn_R-50-FPN_giou_1x.yaml  # Faster R-CNN + GIoU loss
e2e_faster_rcnn_R-50-FPN_iou_1x.yaml   # Faster R-CNN + IoU loss
e2e_mask_rcnn_R-50-FPN_giou_1x.yaml    # Mask R-CNN + GIoU loss
e2e_mask_rcnn_R-50-FPN_iou_1x.yaml     # Mask R-CNN + IoU loss
```

## Train and evaluation commands

For detailed installation instruction and network training options, please take a look at the README file or issue of [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch). Following is a sample command we used for training and testing Faster R-CNN with GIoU.

```
python tools/train_net_step.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_giou_1x.yaml --use_tfboard
python tools/test_net.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_giou_1x.yaml --load_ckpt {full_path_of_the_trained_weight}
```

## Pretrained weights

Here are the trained models using the configurations in this repository.

 - [Faster RCNN + SmoothL1](https://giou.stanford.edu/rcnn_weights/faster_sl1.pth)
 - [Faster RCNN + IoU](https://giou.stanford.edu/rcnn_weights/faster_iou.pth)
 - [Faster RCNN + RPN IoU loss + IoU](https://giou.stanford.edu/rcnn_weights/faster_rpn_iou.pth)
 - [Faster RCNN + GIoU](https://giou.stanford.edu/rcnn_weights/faster_giou.pth)
 - [Faster RCNN + RPN GIoU loss + GIoU](https://giou.stanford.edu/rcnn_weights/faster_rpn_giou.pth)
 - [Mask RCNN + SmoothL1](https://giou.stanford.edu/rcnn_weights/mask_sl1.pth)
 - [Mask RCNN + IoU](https://giou.stanford.edu/rcnn_weights/mask_iou.pth)
 - [Mask RCNN + GIoU](https://giou.stanford.edu/rcnn_weights/mask_giou.pth)
