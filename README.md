# Multimodal Contextual Interactions of Entities: A Modality Circular Fusion Approach for Link Prediction

## Overview
<p align="center">
  <img src="model.jpg" alt="MoCi" width="1000">
</p>
The framework of our method, MoCi, considers multimodal information (𝑠, 𝑣, 𝑡). Initial features are derived from modality-specific embeddings initialization, followed by contrastive pre-training. Pre-aligned semantic features are processed through the modality circular fusion module. Then, these modality features and joint features are input into the relational context-aware prediction module.

## Requirments
- Python==3.8
- numpy==1.24.3
- pillow==10.2.0
- torch==1.13.0
- tqdm==4.66.1
- wheel==0.41.2

## 🎆 Code Structure
The code is organized as follows:
```text
├── datasets
│   ├── DB15K
│   ├── MKG-W
│   └── MKG-Y
│   └── VTJG-I
│   └── VTKG-C
│   └── WN18RR++
│   └── YAGO15K
├── layers
│   ├── __init__
│   └── layer
├── models
│   ├── __init__
│   └── model
├── utils
│   ├── __init__
│   ├── data_loader
│   └── data_util
├── Readme.md
└── main.py
```

## Implementation Details
Our experiments were conducted on an NVIDIA RTX A6000 GPU with 48GB of RAM, utilizing the PyTorch deep learning framework for implementation. Throughout the training process, we configured the number of training epochs to 1,000, with a batch size of 256, modality embedding dimensions set at 256, and a learning rate of 0.0005. 

## Usage

Model Training

    python main.py 


##  🤝 Citation
```python
@inproceedings{
  title={},
  author={},
  booktitle={},
  pages={},
  year={}

}
```