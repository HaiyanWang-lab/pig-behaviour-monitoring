StrongSORT-YOLOv11 for Pig Behaviour Monitoring
================================================
An optimized multi-object tracking (MOT) framework based on StrongSORT and YOLOv11 for precision livestock farming, focusing on real-time, high-accuracy pig behaviour monitoring.

This project adapts the StrongSORT tracking algorithm to the pig behaviour monitoring scenario, integrating YOLOv11 for pig detection and optimizing the tracking pipeline to address farm-specific challenges (e.g., occlusion, low light, pig identity switching).

================================================
🛠 Environment Setup
================================================
Create Conda Environment:
------------------------------------------------
# Create and activate environment
conda create -n pig-tracking python=3.8 -y
conda activate pig-tracking

# Install core dependencies (PyTorch with CUDA support)
# Adjust torch version according to your CUDA version (e.g., CUDA 11.8)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
pip install opencv-python scipy scikit-learn==0.19.2
pip install ultralytics  # For YOLOv11
pip install tqdm pandas numpy  # For data processing & visualization

================================================
📊 Dataset Preparation
================================================
1. Pig Behaviour Dataset Structure
------------------------------------------------
Organize your pig monitoring dataset following this structure (compatible with MOTChallenge format):

path_to_dataset/PigBehaviour
├── train  # Training sequences (video + annotations)
│   ├── pen_01
│   │   ├── img1  # Frames (jpg/png)
│   │   ├── det    # Detection results (optional)
│   │   └── gt     # Ground truth annotations (MOT format)
│   └── pen_02
└── test   # Test sequences (only video/frames)
    ├── pen_03
    └── pen_04

2. Dataset Annotation Format
------------------------------------------------
Follow MOTChallenge annotation format for ground truth (gt.txt):

frame_id, pig_id, x1, y1, w, h, conf, -1, -1, -1
# Example: 1, 5, 320, 240, 80, 100, 1, -1, -1, -1

================================================
⚙️ Configuration
================================================
Modify paths and hyperparameters in opts.py to match your setup:
------------------------------------------------
# Dataset paths
root_dataset = "path_to_dataset/PigBehaviour"  # Your pig dataset root
path_AFLink = "path_to_files/AFLink_epoch20_pig.pth"  # Pig-specific AFLink
dir_dets = "path_to_files/Pig_detections+features"  # Pre-computed detections
path_ECC = "path_to_files/Pig_ECC.json"  # Pig ECC model
dir_save = "results/pig_tracking"  # Save tracking results

# Pig tracking hyperparameters (optimized for farm scenarios)
CONF_THRESH = 0.45  # Lower than default for small pigs
IOU_THRESH = 0.3    # Adjust for pig occlusion
MAX_AGE = 30        # Longer for temporary pig occlusion
NN_BUDGET = 100     # Appearance feature budget
