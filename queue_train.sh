#!/bin/bash
source /cvgl2/u/mihirp/GIoU/Detectron.pytorch/venv/bin/activate
source /cvgl2/u/mihirp/JRDB/detectron2/vars.sh

# Normal Training
python tools/train_net_step.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_1x.yaml --use_tfboard

# IoU Training
# python tools/train_net_step.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_iou_1x.yaml --use_tfboard

# GIoU Training
# python tools/train_net_step.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_giou_1x.yaml --use_tfboard

# GIoU++ Training
# python tools/train_net_step.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_gioupp_1x.yaml --use_tfboard

