# Face Embeddings

[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.4-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.4-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)

This repository integrates and extends the code from <https://github.com/jonasgrebe/pt-femb-face-embeddings>.

The datasets for training can be downloaded from the [Insightface Datasets Github Page](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).
The training datasets are provided in mxnet RecordIO file format.

## Set up repository

We recommend [miniforge](https://conda-forge.org/download/) to set up your python environment.
In case VSCode does not detect your conda environments, install [nb_conda](https://github.com/conda-forge/nb_conda-feedstock) in the base environment.

```bash
conda env create -n $YOUR_ENV_NAME -f environment.yml
conda activate $YOUR_ENV_NAME
pip install -r requirements.txt
pre-commit install
```

## Prepare datasets

To use the training datasets in the current python environment, we require the datasets in the Huggingface datasets format.
Therefore we suggest to use the `mxnet2hf.py` script from the following repository: <https://github.com/alexkrz/dataset-container>.

We use the XQLFW dataset as default for evaluation.
The `xqlfw_aligned_112.zip` file can be downloaded from here: <https://martlgap.github.io/xqlfw/pages/download.html>

## Run training and inference

We provide a train script and a predict script that can be executed with a matching config file.

Example for training:

```bash
python train.py --config configs/train_arcface_ddp.yaml
```

Example for prediction:

```bash
python predict.py --config configs/predict_arcface.yaml
```

To generate predictions on the official model checkpoints, you first need to download the checkpoints and put them into the `checkpoints/` directory.

Checkpoints for ArcFace can be downloaded at: [Arcface Repository](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)

Checkpoints for MagFace can be downloaded at: [MagFace Repository](https://github.com/IrvingMeng/MagFace?tab=readme-ov-file)

## Model references

### Backbone

- IResNet: [Improved Residual Networks for Image and Video Recognition](https://ieeexplore.ieee.org/document/9412193) (ICCV, 2021)

### Headers

- SphereFaceHeader: [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://ieeexplore.ieee.org/document/8100196) (CVPR, 2017)
- CosFaceHeader: [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://ieeexplore.ieee.org/document/8578650) (CVPR, 2018)
- ArcFaceHeader: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://ieeexplore.ieee.org/document/8953658) (CVPR, 2019)
- MagFaceHeader: [MagFace: A Universal Representation for Face Recognition and Quality Assessment](https://ieeexplore.ieee.org/document/9578764) (CVPR, 2021)

## Todos

- [x] Load ArcFace checkpoint from official Arcface repository
- [x] Train own ArcFace model and save it compatible to official Arcface checkpoint
- [x] Add ElasticFace header
- [ ] Compare MagFace training to official Magface code
- [x] Remove `mxnet`dependency. Therefore it is necessary to convert the datasets.
