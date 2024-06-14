# Multimodal Contextual Interactions of Entities: A Modality Circular Fusion Approach for Link Prediction

## Overview
<p align="center">
  <img src="model.jpg" alt="MoCi" width="1000">
</p>

## Requirments
- Python==3.8
- numpy==1.24.3
- pillow==10.2.0
- torch==1.13.0
- tqdm==4.66.1
- wheel==0.41.2

## Code Structure
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

## Usage
    
Model Training

    python main.py 