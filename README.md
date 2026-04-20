# StrongSORT-YOLOv11 for Pig Behaviour Monitoring
An optimized multi-object tracking framework for real-time pig behaviour monitoring in precision livestock farming.

## Abstract
This project adapts StrongSORT and YOLOv11 for pig monitoring tasks, focusing on occlusion, low light, and identity switching in real farm environments.


## Environment Setup
Create a Conda environment:
conda create -n pig-tracking python=3.8 -y
conda activate pig-tracking
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python scipy scikit-learn==0.19.2
pip install ultralytics tqdm pandas numpy

## Pig_Behavior_Dataset
This project is used to publish the pig behavior dataset collected and annotated by our research team for behavior recognition, behavior detection, and individual tracking tasks in group-housed environments.

The dataset contains original surveillance videos and corresponding annotations, including **individual tracking** and **pig multi-behavior detection** labels.

The title of the related paper: Innovative Dual-Network Framework for Monitoring Multiple Behaviors of Individual Pigs in Group-Housed Environments

Dataset Link: https://pan.quark.cn/s/b253138556fd

If you want to use our dataset, please contact us by email to obtain the extraction code or further information.

Contact Email: shiwenhui@webmail.hzau.edu.cn; cjs@webmail.hzau.edu.cn

Dataset Structure 
```text
Pig_Behavior_Dataset/
├── Original_video/
│   ├── D06_20210907070027_min1.mp4
│   ├── D06_20210907070027_min2.mp4"
│   ├── D06_20210907070027_min3.mp4"
│   └── ...
└── annotations/
    ├── individual_tracking/
    │   ├── labels/
    └── pig_multi_behavior_detection/
        ├── labels/
