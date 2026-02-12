
from ultralytics.models.yolo.detect import DetectionValidator

args = dict(model="runs/detect/train/weights/best.pt", data="coco.yaml")
validator = DetectionValidator(args=args)
validator()