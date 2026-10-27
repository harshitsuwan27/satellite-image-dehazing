# Dehazing-Satellite-Images
An end-to-end deep learning and ViT framework designed to remove atmospheric haze and fog from satellite imagery to restore clear visibility and improve downstream remote sensing tasks.

# DATASET
Dataset can be downloaded from https://www.kaggle.com/datasets/mohit3430/haze1k

## TO RUN THE CODE:
# PROJECT DIRECTORY

```text
Dehazing-Satellite-Images/
│
├── train.py
├── evaluate.py
├── best_model.pth
└── Distributed_haze1k/
    ├── train/
    │   ├── input/
    │   └── target/
    │
    ├── test_thin/
    │   ├── input/
    │   └── target/
    │
    ├── test_moderate/
    │   ├── input/
    │   └── target/
    │
    └── test_thick/
        ├── input/
        └── target/
```