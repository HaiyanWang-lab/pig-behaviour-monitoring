import os
import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QPainter
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtCore

from main import Main_Window
from ui.login_ui import Ui_MainWindow as loginWindow

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
envpath = '/home/dzy/miniconda3/lib/python3.9/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

class Login_Window(QtWidgets.QMainWindow, loginWindow):

    def __init__(self):
        super(Login_Window, self).__init__()
        self.setFixedSize(1015, 628)
        self.setupUi(self)
       # self.setStyleSheet("background-color: transparent;")
        self.pushButton_oridinary.clicked.connect(self.btn_login_oridinary)
        self.lineEdit_password.setEchoMode(2)
        self.lineEdit_password.setText("admin")
        self.lineEdit_username.setText("admin")

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap("ui/bg.jpg")
        painter.drawPixmap(self.rect(), pixmap)


    def resetInput(self):
        self.lineEdit_username.clear()
        self.lineEdit_password.clear()

    def btn_login_oridinary(self):
        username = self.lineEdit_username.text()
        password = self.lineEdit_password.text()
        if username == "admin" and password == "admin":
            detectMainWindow.show()
            self.close()
        else:
            QMessageBox.information(self, "提示", "用户名或密码错误!", QMessageBox.Yes, QMessageBox.Yes)


def windowStyle(tw,title):
    tw.setWindowTitle(title)
    desktop = QDesktopWidget().screenGeometry()
    tw.setFixedWidth(desktop.width()-300)
    tw.setFixedHeight(desktop.height()-100)
    tw.setWindowFlags(tw.windowFlags() & ~Qt.WindowMaximizeButtonHint)
    tw.move(150,50)
def handle_error_message(type_, context, message):
    print(message)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    QtCore.qInstallMessageHandler(handle_error_message)

    login = Login_Window()
    windowStyle(login,"pig运动分析系统")

    detectMainWindow = Main_Window(login)
    windowStyle(detectMainWindow,"pig运动分析系统")

    login.show()
    sys.exit(app.exec_())

