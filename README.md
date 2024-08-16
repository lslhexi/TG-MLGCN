# TG-MLGCN
## Brief Introduction

MLGCN only supports training of COCO and VOC datasets, and does not provide code to generate adj matrix and word2vec matrix. To make it easier for MLGCN to train on custom datasets, modifications were made based on the MLGCN source code. In addition, the project uses a relatively new environment configuration.

The project is simple, and the starting point is simply to be able to train my own dataset and provide a reference for others with similar needs.

MLGCN只支持COCO与VOC数据集的训练，并且没有提供生成adj矩阵和word2vec矩阵的代码。为了使MLGCN能够更方便地在自定义数据集上进行训练，基于MLGCN源代码进行了复现和修改。此外，项目使用了较新的环境配置。

项目很简单，出发点只是为了能够训练我自己的数据集，并且给相同需求的朋友们一个参考。

## Reference

ML-GCN paper：[Multi-Label Image Recognition with Graph Convolutional Networks](https://arxiv.org/abs/1904.03582), CVPR 2019.

ML-GCN github：https://github.com/megvii-research/ML-GCN

## Getting Started
### Environment

The project provides a successful environment setting as follows.
you can install the following packages:

* python=3.8
* torch=1.12.0+cu116
* numpy=1.22.4
* torchnet=0.0.4
* tqdm=4.66.1
* torchvision=0.13.0+cu116
* scikit-learn=1.3.0

### Options

- `lr`: learning rate
- `lrp`: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is `lr * lrp`
- `batch_size`: number of images per batch
- `image_size`: size of the resize image (Because the image size of my dataset is too small，need to resize)
- `epochs`: number of training epochs
- `checkpoint_interval`: interval of saving .pth
- `log_interval`: interval of log
- `val_interval`: interval of validation
- `MultiStepLR`: lr_decay

### Demo

```sh
cd TG-MLGCN/

python train.py --image_size 448 --epochs 20 --batchsize 16 --lr 0.001 --lrp 0.1 --msl True --checkpoint_interval 5 --log_interval 100 --val_interval 5
```

### Tips
Before you begin training, you need to create some folder and Modifying the file path because I didn't upload my weights.
```sh
-TG_MLGCN
        |-word_embding
                |-word_embding.npy
        |-data
                |-TG1
                        |-anno
                                |-train_no_rpt.json
                                |-test_no_rpt.json
                        |-img
                                |-1.jpg
                                |-2.jpg
        |-glove.6B
                |-resultFile
                        |-glove.6B.300d.txt
        |-checkpoint
                |-TG
```
