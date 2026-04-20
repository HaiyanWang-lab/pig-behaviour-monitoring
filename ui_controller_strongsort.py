import torch

from deep_sort import build_tracker
from ultralytics import YOLO
import argparse
import copy
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

import warnings

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils.draw import draw_boxes
from ultralytics.utils.files import increment_path
from ultralytics.utils.parser import get_config
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.models.yolo.detect import DetectionPredictor
def calculate_iou(box1, box2):
    # 计算交集的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    # 计算交集的面积
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    # 计算两个矩形框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # 计算并集的面积
    union_area = box1_area + box2_area - intersection_area
    # 计算IoU
    if union_area == 0:
        return 0
    return intersection_area / union_area


class Detect():
    def __init__(self):
        args = dict(model="runs/detect/train/weights/best.pt" ,conf=0.4)
        self.predictor = DetectionPredictor(overrides=args)


        args1 = dict(model="runs/detect/pigaction/weights/best.pt" ,conf=0.4)
        self.predictor_action = DetectionPredictor(overrides=args1)


        self.reset()
        ###deepsort
        cfg = get_config()
        cfg.merge_from_file("./deep_sort/deep_sort.yaml")
        self.deepsort = build_tracker(cfg, use_cuda=True)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.pts = [deque(maxlen=100) for _ in range(9999)]



    def reset(self):
        self.alreadyCounter = []
        self.track_history = defaultdict(list)
        self.clsMap = {"0":"standing","1":"side lying","2":"prone lying"}
        self.counter  = {"standing":[], "side lying":[], "prone lying":[]}
        self.id2info = {}
        self.allids = {}

        self.names_action =  {0:"attack",1:"diet",2:"normal"}
        self.counter_action  = {"attack":[], "diet":[], "normal":[] }
        self.id2info_action = {}


    def detect(self,img0, frame_id=1, callback= None):
        origin = img0.copy()
        drawResult = img0.copy()
        self.predictor.predict_cli(img0)
        self.predictor_action.predict_cli(img0)
        self.names = self.predictor.results[0].names
        result = self.predictor.plotted_img
        finalTorch = []


        #  获取攻击还是觅食
        actionboxs =[]
        actioncls =[]
        for boxex in self.predictor_action.results[0].boxes:
            for box,cls,conf in zip(boxex.xyxy,boxex.cls,boxex.conf):
                left, top, right, bottom =[int(x) for x in box]
                actionboxs.append([left,top,right,bottom])
                actioncls.append(self.names_action[int(cls)])

        for boxex in self.predictor.results[0].boxes:
            for box,cls,conf in zip(boxex.xyxy,boxex.cls,boxex.conf):
                left, top, right, bottom =[int(x) for x in box]
                tmp = [(left + right) / 2, (top + bottom) / 2, right - left, bottom - top, float(conf), int(cls)]
                finalTorch.append(tmp)


        if len(finalTorch) > 0:
            bbox_xywh = torch.tensor(finalTorch)[:, :4]
            confs = torch.tensor(finalTorch)[:, 4:5]
            clss = torch.tensor(finalTorch)[:, 5:6]
            outputs = self.deepsort.update(bbox_xywh, confs, clss,origin)  # deepsort
        else:
            outputs = []
        isDraw = True
        if len(outputs) > 0 and isDraw:
            bbox_xyxy = outputs[:, :4]
            clid = outputs[:, 4]
            identities= outputs[:, -1]

            actionshows =[]
            for box , track_id,cid in zip(bbox_xyxy,identities,clid):
                currentAction = ""
                max_iou = 0
                # 得到他是不是在饮食 还是攻击
                for j, (box2,cls1) in enumerate(zip(actionboxs,actioncls)):
                    iou = calculate_iou(box, box2)
                    if iou > max_iou:
                        max_iou = iou
                        currentAction = cls1
                if len(currentAction)<=0:
                    currentAction="normal"
                actionshows.append(currentAction)

                realname = self.clsMap[str(cid)]
                if track_id not in self.allids.keys():
                    self.allids[track_id] = 0
                if track_id not in self.id2info.keys():
                    self.id2info[track_id] = {"standing":[], "side lying":[], "prone lying":[]}
                    self.id2info_action[track_id] = {"attack":[], "diet":[], "normal":[] }
                self.id2info[track_id][realname].append(frame_id)
                self.id2info_action[track_id][currentAction].append(frame_id)

                self.allids[track_id]+=1
                if  str(track_id) not in  self.counter[realname] and  self.allids[track_id]>20:
                    self.counter[realname].append(str(track_id))

                if  str(track_id) not in  self.counter_action[currentAction] and  self.allids[track_id]>20:
                    self.counter_action[currentAction].append(str(track_id))


            drawResult = draw_boxes(drawResult, bbox_xyxy, identities,clid=clid,acions=actionshows,names = self.clsMap,pts=self.pts)

        if callback is not None:
             callback(drawResult ,False)
        return drawResult,self.counter

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
    cap = cv2.VideoCapture("./08.mp4")
    # 获取原始视频的帧率、宽度和高度
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale_factor = 0.5  # 可以根据需要调整这个比例，例如 0.5 表示缩小到原来的一半
    width = int(original_width * scale_factor)
    height = int(original_height * scale_factor)

    # 创建 VideoWriter 对象，用于保存结果视频
    out = cv2.VideoWriter('attack_diet_test_output.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 800, 800)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            out.release()
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        img,info = detect.detect(frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 255)
        thickness = 2
        # y = 0
        # for item in detect.counter.items():
        #     y+=50
        #     line = item[0] + f": {len(item[1])} ids: " + ",".join([str(x) for x in item[1]])
        #     cv2.putText(img, line, (50, y), font, font_scale, color, thickness)
        #
        #
        # for item in detect.counter_action.items():
        #     if item[0] == "normal":
        #         continue
        #     y+=50
        #     line = item[0] + f": {len(item[1])} ids: " + ",".join([str(x) for x in item[1]])
        #     cv2.putText(img, line, (50, y), font, font_scale, color, thickness)


        cv2.imshow('Image', img)
        out.write(img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        #time.sleep(1)
    out.release()
    cap.release()
    cv2.destroyAllWindows()
