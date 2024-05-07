# Face Embeddings

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![pytorch](https://img.shields.io/badge/PyTorch_1.13-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.8.6-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)

This repository integrates and extends the code from <https://github.com/jonasgrebe/pt-femb-face-embeddings>.

The datasets for training can be downloaded from the [Insightface Datasets Github Page](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).
The training datasets are provided in mxnet RecordIO file format.

## Set up repository

1. Install miniconda
2. Create environment with

```bash
conda env create -n $YOUR_ENV_NAME -f environment.yml
```

3. Install pip requirements with

```bash
pip install -r requirements.txt
```

4. Install pre-commit hooks with

```bash
pre-commit install
```

## Model references

### Backbone:

- IResNet: [Improved Residual Networks for Image and Video Recognition](https://ieeexplore.ieee.org/document/9412193) (ICCV, 2021)

### Headers:

- SphereFaceHeader: [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://ieeexplore.ieee.org/document/8100196) (CVPR, 2017)
- CosFaceHeader: [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://ieeexplore.ieee.org/document/8578650) (CVPR, 2018)
- ArcFaceHeader: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://ieeexplore.ieee.org/document/8953658) (CVPR, 2019)
- MagFaceHeader: [MagFace: A Universal Representation for Face Recognition and Quality Assessment](https://ieeexplore.ieee.org/document/9578764) (CVPR, 2021)

## Todos

- [ ] Try to load ArcFace checkpoint from official Arcface repository
- [ ] Add ElasticFace header
