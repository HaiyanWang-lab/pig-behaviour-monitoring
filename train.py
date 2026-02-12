from ultralytics.models.yolo.detect import DetectionTrainer
import torch
torch.cuda.empty_cache() 
args = dict(model="yolo11s.yaml", data="coco.yaml", epochs=130,batch=15)
trainer = DetectionTrainer(overrides=args)
trainer.train()                                       