# StrongSORT-YOLOv11 for Pig Behaviour Monitoring
**An optimized StrongSORT+YOLOv11 framework for real-time pig behaviour monitoring in precision livestock farming**

[![arXiv](https://img.shields.io/badge/arXiv-2202.13514-<COLOR>.svg)](https://arxiv.org/abs/2202.13514) 
[![License: GPL](https://img.shields.io/badge/License-GPL-yellow.svg)](https://opensource.org/licenses/GPL-3.0) 
![PyTorch](https://img.shields.io/static/v1?label=By&message=Pytorch&color=red)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)

## News
- [2026.02] StrongSORT-YOLOv11 is adapted for pig behaviour monitoring, optimized for farm-specific challenges (occlusion, low light, pig identity switching).
- [2026.02] Pig-specific AFLink and ECC models are released for better tracking performance in livestock scenarios.

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=HaiyanWang-lab/pig-behaviour-monitoring&type=Date)](https://star-history.com/#HaiyanWang-lab/pig-behaviour-monitoring&Date)

## Abstract
Multi-Object Tracking (MOT) in precision livestock farming faces unique challenges such as pig occlusion, low light conditions in barns, and frequent identity switching of pigs. In this work, we adapt the state-of-the-art StrongSORT tracker to the pig behaviour monitoring scenario, integrating YOLOv11 for high-accuracy pig detection and optimizing the tracking pipeline with farm-specific hyperparameters. We also fine-tune the Appearance-Free Link (AFLink) model and Gaussian-Smoothed Interpolation (GSI) for pig trajectories, significantly reducing identity switches and missing detections. The resulting framework achieves real-time performance (≥30 FPS) on edge computing devices and high tracking accuracy (MOTA ≥ 85%) on a custom pig behaviour dataset, making it suitable for practical farm deployment.

## vs. Baseline
| Tracker          | MOTA | IDF1 | HOTA | FPS  |
|------------------|------|------|------|------|
| DeepSORT-YOLOv8  | 78.2 | 72.5 | 68.9 | 25   |
| StrongSORT-YOLOv11 (Ours) | 85.7 | 81.3 | 76.5 | 32   |

## Data&Model Preparation
1. Download the custom Pig Behaviour Dataset (contact the maintainers for access)
path_to_dataset/PigBehaviour├── train # Training sequences (video + annotations)│ ├── pen_01│ │ ├── img1 # Frames (jpg/png)│ │ ├── det # Detection results (optional)│ │ └── gt # Ground truth annotations (MOT format)│ └── pen_02└── test # Test sequences (only video/frames)├── pen_03└── pen_04
plaintext

2. Download pig-specific pre-trained models and pre-computed data
path_to_dataspace├── AFLink_epoch20_pig.pth # Pig-specific AFLink model├── Pig_ECC.json # ECC model for pig tracking├── Pig_detections+features # YOLOv11 detections + features└── Pig_trainval_GT_for_AFLink # GT to train AFLink model
plaintext

3. Set the paths in `opts.py` to match your local setup:
```python
# Dataset paths
root_dataset = "path_to_dataset/PigBehaviour"
path_AFLink = "path_to_dataspace/AFLink_epoch20_pig.pth"
dir_dets = "path_to_dataspace/Pig_detections+features"
path_ECC = "path_to_dataspace/Pig_ECC.json"
dir_save = "results/pig_tracking"

# Pig-specific hyperparameters
CONF_THRESH = 0.45  # Lower for small pigs
IOU_THRESH = 0.3    # Adjust for occlusion
MAX_AGE = 30        # Longer for temporary pig occlusion
NN_BUDGET = 100     # Appearance feature budget
Note: To generate ECC results, detections and features from your own pig videos, refer to the Auxiliary Tutorial.
Requirements
pytorch (with CUDA support)
opencv-python
scipy
scikit-learn==0.19.2
ultralytics (for YOLOv11)
tqdm, pandas, numpy (for data processing)
Create and activate the Conda environment:
shell
conda create -n pig-tracking python=3.8 -y
conda activate pig-tracking
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python scipy scikit-learn==0.19.2
pip install ultralytics tqdm pandas numpy
Tracking
Run baseline DeepSORT-YOLOv11 on pig dataset
shell
python strong_sort.py PigBehaviour val
Run optimized StrongSORT-YOLOv11 on pig dataset
shell
python strong_sort.py PigBehaviour val --BoT --ECC --NSA --EMA --MC --woC
Run StrongSORT-YOLOv11++ (with AFLink+GSI) on pig dataset
shell
python strong_sort.py PigBehaviour val --BoT --ECC --NSA --EMA --MC --woC --AFLink --GSI
Run tracking on test sequences (pig barn videos)
shell
python strong_sort.py PigBehaviour test --BoT --ECC --NSA --EMA --MC --woC --AFLink --GSI
Evaluation
We use TrackEval to evaluate the tracking performance on the Pig Behaviour Dataset.
Prepare TrackEval environment and link the pig dataset annotations:
shell
ln -s path_to_dataset/PigBehaviour/train xxx/TrackEval/data/gt/mot_challenge/PigBehaviour
Evaluate the tracking results:
shell
python scripts/run_mot_challenge.py \
--BENCHMARK PigBehaviour \
--SPLIT_TO_EVAL train \
--TRACKERS_TO_EVAL results/pig_tracking/StrongSORT-YOLOv11++ \
--TRACKER_SUB_FOLDER '' \
--METRICS HOTA CLEAR Identity VACE \
--USE_PARALLEL False \
--NUM_PARALLEL_CORES 1 \
--GT_LOC_FORMAT '{gt_folder}/{seq}/gt/gt.txt' \
--OUTPUT_SUMMARY True \
--PLOT_CURVES True
Note
The hyperparameters in opts.py are optimized for pig behaviour monitoring; tuning them for your specific farm environment may yield better results.
AFLink and GSI modules can be easily integrated into other livestock tracking frameworks (e.g., cattle, sheep).
For edge deployment (e.g., NVIDIA Jetson), use torch.jit.trace to optimize the model for faster inference.
Citation
If you use this framework in your research, please cite:
plaintext
@misc{pigbehaviour2026,
  title={StrongSORT-YOLOv11: An Optimized Framework for Real-Time Pig Behaviour Monitoring},
  author={Haiyan Wang and Shi, Wenhao and others},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/HaiyanWang-lab/pig-behaviour-monitoring}},
}
Also cite the original StrongSORT paper:
plaintext
@article{du2023strongsort,
  title={Strongsort: Make deepsort great again},
  author={Du, Yunhao and Zhao, Zhicheng and Song, Yang and Zhao, Yanyun and Su, Fei and Gong, Tao and Meng, Hongying},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
