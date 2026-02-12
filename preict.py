
from ultralytics.utils import ASSETS
from ultralytics.models.yolo.detect import DetectionPredictor

args = dict(model="runs/detect/train/weights/best.pt", source="/home/wangxp/pig_data_all/Posture_7220_my_yolo_dataset/val/images/")
predictor = DetectionPredictor(overrides=args)
predictor.predict_cli()