# StrongSORT-YOLOv11 for Pig Behaviour Monitoring
An optimized multi-object tracking framework for real-time pig behaviour monitoring in precision livestock farming.

## Abstract
This project adapts StrongSORT and YOLOv11 for pig monitoring tasks, focusing on occlusion, low light, and identity switching in real farm environments.

## Environment Setup
Create a Conda environment:
```bash
conda create -n pig-tracking python=3.8 -y
conda activate pig-tracking
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python scipy scikit-learn==0.19.2
pip install ultralytics tqdm pandas numpy

## 
