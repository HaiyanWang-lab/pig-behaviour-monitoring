# StrongSORT-YOLOv11 for Pig Behaviour Monitoring
An optimized multi-object tracking framework for real-time pig behaviour monitoring in precision livestock farming.

## Abstract
This project adapts StrongSORT and YOLOv11 for pig monitoring tasks, focusing on occlusion, low light, and identity switching in real farm environments.


# Pig_Behavior_Dataset

## overview

![Uploading image.png…]()

Due to GitHub's storage capacity limitations, the comprehensive dataset for this project including the raw surveillance video footage and the corresponding tracking and detection annotations is hosted on a cloud drive.

This dataset is developed for individual pig behavior monitoring, serving as a critical resource for an optimized multi-object tracking framework tailored for real-time applications in precision livestock farming.

The collection comprises raw surveillance video footage recorded in a commercial group-housed environment, accompanied by meticulous frame-level annotations. These annotations support both continuous individual identity tracking (unique target IDs) and multi-behavior recognition across five primary categories: standing, side-lying, prone lying, climbing, and feeding. By providing high-quality spatio-temporal ground truth, this dataset aims to facilitate the development of robust models capable of handling complex farm scenarios—such as severe occlusion and frequent identity switching—thereby advancing automated animal welfare assessment and precision breeding technologies.

This project publishes a pig behavior dataset collected and annotated by our research team.  
It is designed for:

- Behavior recognition  
- Behavior detection  
- Multi-object tracking  
- Individual-level analysis in group-housed environments  

The dataset contains original surveillance videos and corresponding annotations, including:

- Individual tracking (ID-based annotations)  
- Pig multi-behavior detection labels  

---

## Related Paper

**Innovative Dual-Network Framework for Monitoring Multiple Behaviors of Individual Pigs in Group-Housed Environments**

---

## Dataset Download

Link: https://pan.quark.cn/s/b253138556fd  Extraction code: Please send an email to the author

> Please contact the authors to obtain the extraction code.

---

## Contact

- shiwenhui@webmail.hzau.edu.cn  
- cjs@webmail.hzau.edu.cn  

---

## Dataset Structure

```text
Pig_Behavior_Dataset/
├── Original_video/
│   ├── D06_20210907070027_min1.mp4
│   ├── D06_20210907070027_min2.mp4
│   ├── D06_20210907070027_min3.mp4
│   └── ...
│
└── annotations/
    ├── individual_tracking/
    │   ├── labels/
    │   │   ├── 000001.txt
    │   │   ├── 000002.txt
    │   │   └── ...
    │
    └── pig_multi_behavior_detection/
        ├── labels/
        │   ├── 000001.txt
        │   ├── 000002.txt
        │   └── ...
