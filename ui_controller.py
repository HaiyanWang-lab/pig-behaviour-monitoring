
from ultralytics import YOLO
import argparse
import copy
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

import warnings

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors


class Detect():
    def __init__(self):
        self.model = YOLO("runs/detect/train/weights/best.pt" )
        device="0"
        self.model.to("cuda") if device == "0" else self.model.to("cpu")
        self.track_history = defaultdict(list)
        self.names = self.model.model.names
        self.alreadyCounter = []


    def reset(self):
        self.alreadyCounter = []
        self.track_history = defaultdict(list)

    def detect(self,img0, callback= None):
        line_thickness=2
        track_thickness=2
        results = self.model.track(img0, persist=True, classes=[0,2], conf=0.1)
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            annotator = Annotator(img0, line_width=line_thickness, example=str(self.names ))
            for box, track_id, cls,conf in zip(boxes, track_ids, clss,confs):
                annotator.box_label(box, str(self.names[cls])+ "_"+str(track_id), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center
                #img0 = annotator.im
                track = self.track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(img0, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

        if callback is not None:
             callback(img0 )
        return img0



    def detectPic(self,img0, callback= None):
        line_thickness=2
        results = self.model.predict(img0, )
        isneerecord = False
        if results[0].boxes  is not None:
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            annotator = Annotator(img0, line_width=line_thickness, example=str(self.names ))
            for box,  cls,conf in zip(boxes,  clss,confs):
                annotator.box_label(box, str(self.names[cls]) , color=colors(cls, True))
        if callback is not None:
            callback(img0,str(0) ,str(0), isneerecord,False)

if __name__=="__main__":
    detect = Detect()
    cap = cv2.VideoCapture("./output_video.mp4")

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 800, 800)

    while cap.isOpened():
        ret, frame = cap.read()
        img = detect.detect(frame)
        if not ret:
            break
        cv2.imshow('Image', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        time.sleep(1)
    cap.release()
    cv2.destroyAllWindows()