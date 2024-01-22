## Learning discriminative foreground-and-background features for few-shot segmentation
This is the official implementation of the paper "Learning discriminative foreground-and-background features for few-shot segmentation".

## Requirements

- Python 3.7
- PyTorch 1.5.1
- cuda 10.1
- tensorboard 1.14

Conda environment settings:

```bash
conda create -n DFBnet python=3.7
conda activate DFBNet

conda install pytorch=1.5.1 torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge tensorflow
pip install tensorboardX
```

## Prepare Datasets

Download COCO2014 train/val images and annotations: 

```bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```

Download COCO2014 train/val annotations from Google Drive: [[train2014.zip](https://drive.google.com/file/d/1fcwqp0eQ_Ngf-8ZE73EsHKP8ZLfORdWR/view?usp=sharing)], [[val2014.zip](https://drive.google.com/file/d/16IJeYqt9oHbqnSI9m2nTXcxQWNXCfiGb/view?usp=sharing)].(and locate both train2014/ and val2014/ under annotations/ directory).

Create a directory 'datasets' and appropriately place coco to have following directory structure:

    datasets/
        └── COCO2014/           
            ├── annotations/
            │   ├── train2014/  # (dir.) training masks (from Google Drive) 
            │   ├── val2014/    # (dir.) validation masks (from Google Drive)
            │   └── ..some json files..
            ├── train2014/
            └── val2014/

## Prepare backbones

Downloading the following pre-trained backbones:

> 1. [ResNet-50](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1h-35c100f8.pth) pretrained on ImageNet-1K by [TIMM](https://github.com/rwightman/pytorch-image-models)
> 2. [ResNet-101](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth) pretrained on ImageNet-1K by [TIMM](https://github.com/rwightman/pytorch-image-models)

Create a directory 'backbones' to place the above backbones. The overall directory structure should be like this:

    ../                         # parent directory
    ├── DFBNet/                 # current (project) directory
    │   ├── common/             # (dir.) helper functions
    │   ├── data/               # (dir.) dataloaders and splits for each FSS dataset
    │   ├── model/              # (dir.) implementation of DFBNet
    │   ├── README.md           # intstruction for reproduction
    │   ├── train.py            # code for training
    │   ├── test.py             # code for testing
    │   ├── train.sh            # script for training
    │   └── train.sh            # script for testing
    ├── datasets/               # (dir.) Few-Shot Segmentation Datasets
    └── backbones/              # (dir.) Pre-trained backbones

## Train and Test
You can use our scripts to build your own. For more information, please refer to ./common/config.py

> ```bash
> sh ./train.sh
> ```
> 
> - For each experiment, a directory that logs training progress will be automatically generated under logs/ directory. 
> - From terminal, run 'tensorboard --logdir logs/' to monitor the training progress.
> - Choose the best model when the validation (mIoU) curve starts to saturate. 

For testing, you have to prepare a pretrained model. The pretrained model of DFBNet will be released soon. 
> ```bash
> sh test.sh
> ```
> 



## BibTeX
If you are interested in our paper, please cite:
```
@article{jiang2023learning,
  title={Learning discriminative foreground-and-background features for few-shot segmentation},
  author={Jiang, Cong and Zhou, Yange and Liu, Zhaoshuo and Feng, Chaolu and Li, Wei and Yang, Jinzhu},
  journal={Multimedia Tools and Applications},
  pages={1--21},
  year={2023},
  publisher={Springer}
}
```