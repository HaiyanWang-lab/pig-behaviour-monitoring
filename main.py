import os.path
import sys
import threading
from datetime import datetime

import cv2
from PyQt5.QtGui import QImage, QPixmap, QStandardItemModel, QStandardItem, QTextOption, QHelpEvent
from PyQt5.QtWidgets import QAbstractItemView, QTableWidgetItem, QFileDialog, QMessageBox

from createdb import PigDB
from ui import alertdialog
from ui.main_ui import Ui_MainWindow as mainWindow
import numpy as np
import time

from ui_controller_deepsort import Detect

from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from PyQt5.QtGui import QPixmap, QPalette, QBrush
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
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

envpath = '/home/dzy/miniconda3/lib/python3.9/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

warnings.filterwarnings('ignore' )

tmp = []
class CenteredItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        option.displayAlignment = Qt.AlignCenter
        super().paint(painter, option, index)

class TooltipDelegate(QStyledItemDelegate):
    def helpEvent(self, event, view, option, index):
        if isinstance(event, QHelpEvent) and event.type() == QEvent.ToolTip:
            tooltip_text = index.data(Qt.ToolTipRole)
            if tooltip_text:
                QToolTip.showText(event.globalPos(), tooltip_text)
                return True
        return super().helpEvent(event, view, option, index)
class Main_Window(QtWidgets.QMainWindow, mainWindow):
    signal2 = pyqtSignal(str,str)
    def __init__(self,login):
        super(Main_Window, self).__init__()
        self.setupUi(self)
        self.isStoped = False
        self.login = login
        self.ispic = False

        self.pigDb = PigDB()
        self.pigDb.createDb()
        self.pigDb.createTable()
        self.fps = 0

        self.initbtn()
        self.hasInit = False
        self.currentImg = None
        self.instance = Detect()

        self.jinerDialog = QDialog()
        self.myJinErDialog = alertdialog.Ui_Dialog()
        self.myJinErDialog.setupUi(self.jinerDialog)

        self.myJinErDialog.tableView.setSelectionBehavior(QTableView.SelectRows)
        self.myJinErDialog.tableView.setEditTriggers(QTableView.NoEditTriggers)

       # self.myJinErDialog.tableView.horizontalHeader().resize(0, 500)
        self.myJinErDialog.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.signal2.connect(self.signalcall_jiner)
        self.myJinErDialog.tableView.horizontalHeader().setStyleSheet("QHeaderView::section {"
                                                         "background-color: rgb(88, 155, 255);color: rgb(255, 255, 255);}");
        self.myJinErDialog.tableView.verticalHeader().hide()


        self.videoselect = "全部"
        self.idselect = "全部"

        self.myJinErDialog.comboBox_video.currentIndexChanged.connect(self.on_combobox_video_changed)
        self.myJinErDialog.comboBox_id.currentIndexChanged.connect(self.on_combobox_id_changed)

    def initbtn(self):
        #self.pushButton_pic.clicked.connect(self.pic_select)
        self.pushButton_video.clicked.connect(self.video_select)
        self.pushButton_camera.clicked.connect(self.camera_select)
        self.pushButton_stop.clicked.connect(self.stop_detect)
        self.pushButton_detail.clicked.connect(self.detail_detect)

    def on_combobox_video_changed(self, index):
        self.videoselect = self.myJinErDialog.comboBox_video.currentText()
        self.signalcall_jiner()
    def on_combobox_id_changed(self, index):
        self.idselect = self.myJinErDialog.comboBox_id.currentText()
        self.signalcall_jiner()

    def pic_select(self):
        tmp, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "All Files(*);;*.png;;*.jpg;;")
        if tmp is None or tmp == "":
            return
        self.imgName = tmp
        self.ispic = True
        self.startDetect(self.instance.detectPic,True)

    def video_select(self):
        tmp, imgType = QFileDialog.getOpenFileName(self, "打开视频", "", " All Files(*)")
        if tmp is None or tmp == "":
            return
        self.imgName = tmp
        self.ispic = False
        self.instance.reset()
        self.startDetect(self.getvideo,False)

    def camera_select(self):
        self.imgName = 0
        self.ispic = False
        self.instance.reset()
        self.startDetect(self.getvideo,False)

    def startDetect(self,target,isPic):
        self.isStoped= False
        if not isPic:
            self.detectThread = threading.Thread(target=target, args=(self.imgName,self.showImage))
        else:
            self.detectThread = threading.Thread(target=target,args=(cv2.imread(self.imgName),self.showImage))
        self.detectThread.start()

    def getvideo(self,img,callback):
        cap = cv2.VideoCapture(img )
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        vid_frame_count = 0
        while not self.isStoped:
            ret, frame = cap.read()
            if not ret:
                self.stop_detect()
                cap.release()
                break
            current_frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
           # current_time_in_seconds = (current_frame_index / self.fps)
           # print("ddddd"  , current_frame_index , current_time_in_seconds)
            self.instance.detect(frame,current_frame_index,callback)
    '''
    一个列表中保存视频的帧id，从这个列表中，按照帧的不连续状态划分，
    把连续的帧的开始和结束对应的视频帧id放到一个新的list中，
    同时得到每个连续帧序列的帧数，放到一个新列表中
    '''
    def group_continuous_frames(self,frame_ids):
        if not frame_ids:
            return [], []
        result = []
        frame_counts = []
        start = frame_ids[0]
        prev = frame_ids[0]
        count = 1
        for i in range(1, len(frame_ids)):
            if frame_ids[i] == prev + 1:
                prev = frame_ids[i]
                count += 1
            else:
                result.append([start, prev])
                frame_counts.append(count)
                start = frame_ids[i]
                prev = frame_ids[i]
                count = 1
        result.append([start, prev])
        frame_counts.append(count)
        return result, frame_counts


    def stop_detect(self):
        if self.fps == 0:
            return
        finalInfos = {}
        for id, infos in self.instance.id2info.items():
            #print(id)
            finalInfos[id] = {}
            for type,info in infos.items():
                type = type.replace(" ","")
                if type not in finalInfos[id]:
                    finalInfos[id][type] = []
                continuous_frame_ranges, frame_counts = self.group_continuous_frames(info)
                for rang,count in zip(continuous_frame_ranges, frame_counts):
                    if len(rang) == 0 :
                        continue
                    duration = count / self.fps
                    start = (rang[0] / self.fps)
                    end = (rang[1] / self.fps)
                    finalInfos[id][type].append((start,end,duration))
        #print("")
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H_%M_%S")
        for tmp in finalInfos.items():
            self.pigDb.insertInto(videoid = self.imgName+"__"+formatted_datetime,
                                  pigid=str(tmp[0]),
                                  time= str(current_datetime.timestamp()),
                                  stand=str(tmp[1]['standing']),
                                  side=str(tmp[1]['sidelying']),
                                  prone=str(tmp[1]['pronelying']),
                                  videoname = self.imgName.split(os.sep)[-1]) # videoid,pigid, time,stand,side,prone):
        #self.pigDb. in
        self.isStoped= True
        self.imgName = None
        self.fps = 0
        self.instance.reset()

    def detail_detect(self):
        self.showCombox()
        self.signal2.emit("","")
    def showCombox(self):
        self.myJinErDialog.comboBox_video.clear()
        self.myJinErDialog.comboBox_id.clear()

        self.myJinErDialog.comboBox_video.addItem("全部")
        self.myJinErDialog.comboBox_id.addItem("全部")
        allinfos = self.pigDb.selectall()
        allvideos = []
        allids =[]
        for indx,row in enumerate(allinfos):
            allvideos.append(str(row[1]).split(os.sep)[-1].split("__")[0])
            allids.append(str(row[2]))
        allvideos = list(set(allvideos))
        for v in  allvideos:
            self.myJinErDialog.comboBox_video.addItem(v)
        allids = list(set(allids))
        allids = sorted(allids)
        for v in  allids:
            self.myJinErDialog.comboBox_id.addItem(v)

    def signalcall_jiner(self):
        self.centered_delegate = CenteredItemDelegate()
        if self.videoselect == "全部" and self.idselect == "全部":
            allinfos = self.pigDb.selectall()
        else:
            if self.videoselect == "全部":
                allinfos = self.pigDb.selecByPigId(self.idselect)
            elif self.idselect == "全部":
                allinfos = self.pigDb.selecByVideoId(self.videoselect)
            else:
                allinfos = self.pigDb.selecByIds(self.idselect,self.videoselect)

        model = QStandardItemModel(len(allinfos),2)
        self.myJinErDialog.tableView.setModel(model)
       # self.myJinErDialog.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);
        model.setHorizontalHeaderLabels(["时间","ID","站立","站立总时间(s)","侧卧","侧卧总时间(s)","躺着","躺着总时间(s)"])
        for col in range(8):
            self.myJinErDialog.tableView.setItemDelegateForColumn(col, self.centered_delegate)

            delegate = TooltipDelegate()
            self.myJinErDialog.tableView.setItemDelegate(delegate)
        for indx,row in enumerate(allinfos):
            # 时间
            dt_object = datetime.fromtimestamp(eval(row[3]))
            formatted_datetime = dt_object.strftime("%Y-%m-%d %H:%M:%S")
            item = QStandardItem(formatted_datetime)
            model.setItem(indx, 0, item)

            # ID
            item = QStandardItem(row[2])
            model.setItem(indx, 1, item)

            # "站立"
            infos = eval(row[4])
            t = "_".join(str(float(info[0]))+"_"+str(round(float(info[1]),2)) for info in infos)
            item = QStandardItem(t if len(t)>0 else "-")
            model.setItem(indx, 2, item)
            # "站立总时间"
            item = QStandardItem(str(np.sum(np.array([round(float(info[-1]),2) for info in infos]))))
            model.setItem(indx, 3, item)


            # "侧卧"
            infos = eval(row[5])
            t = "_".join(str(float(info[0]))+"_"+str(round(float(info[1]),2)) for info in infos)
            item = QStandardItem(t if len(t)>0 else "-")
            model.setItem(indx, 4, item)
            # "侧卧总时间"
            item = QStandardItem(str(np.sum(np.array([round(float(info[-1]),2) for info in infos]))))
            model.setItem(indx, 5, item)

            # "躺着"
            infos = eval(row[6])
            t = "_".join(str(float(info[0]))+"_"+str(round(float(info[1]),2)) for info in infos)
            item = QStandardItem(t if len(t)>0 else "-")
            model.setItem(indx, 6, item)
            # "躺着总时间"
            item = QStandardItem(str(np.sum(np.array([round(float(info[-1]),2) for info in infos]))))
            model.setItem(indx, 7, item)



        if self.jinerDialog.isVisible():
            print("im showing")
        else:
            self.jinerDialog.show()

    def realShow(self,isInit):
        if self.currentImg is None:
            return
        frame = cv2.cvtColor(self.currentImg, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        width = self.label_2.width()
        height = self.label_2.height()

        scale = max(width, height) / max(frame_w, frame_h)
        frame_h, frame_w = int(frame_h * scale), int(frame_w * scale)
        frame = cv2.cvtColor(self.currentImg, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,(frame_w,frame_h))
        frame = cv2.copyMakeBorder(frame, int(np.maximum((height - frame_h) / 2, 0)),  # up
                                   int(np.maximum((height - frame_h) / 2, 0)),  # down
                                   int(np.maximum((width - frame_w) / 2, 0)),  # left
                                   int(np.maximum((width - frame_w) / 2, 0)),  # right
                                   cv2.BORDER_CONSTANT, value=[88, 155, 255])
        img = QImage(
            frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
        del frame
        self.label_2.setPixmap(QPixmap.fromImage(img))

    def showImage(self,image, isInit):
        # if isneerecord:
        #     self.pigDb.insertInto( time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), personcounter,carcouner)

        self.currentImg = image
        self.realShow(isInit)

    def closeEvent(self, e):
        self.isStoped= True
        self.imgName = None
        self.login.resetInput()
        self.login.show()
        self.instance.reset()
        self.fps = 0

# def handle_error_message(type_, context, message):
#     print(message)
# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     QtCore.qInstallMessageHandler(handle_error_message)
#
#     Ui_Main = Main_Window()
#     Ui_Main.show()
#     sys.exit(app.exec_())
