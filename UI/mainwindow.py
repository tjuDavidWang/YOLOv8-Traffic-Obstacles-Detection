import sys
import threading

import ui_mainwindow
from PyQt6 import QtWidgets
from PyQt6 import QtCore
from PyQt6 import QtGui

import yolo


def start_new_thread_video(signal, args):
    y = yolo.yolo(signal)
    y.detect_video(args)


def start_new_thread_image(signal, args):
    y = yolo.yolo(signal)
    y.detect_image(args)


class MainWindow(QtWidgets.QMainWindow):
    signal = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = ui_mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.btn_useCam.clicked.connect(self.useCam)
        self.ui.btn_useFile.clicked.connect(self.useFile)
        self.ui.btn_useImg.clicked.connect(self.useImg)
        self.ui.btn_exit.clicked.connect(self.exit)
        self.signal.connect(self.draw)

    def useCam(self):
        self.ui.lbl_isRunning.setText("运行中")
        self.ui.lineEdit.setEnabled(False)
        self.ui.lineEdit_2.setEnabled(False)
        self.ui.lineEdit_3.setEnabled(False)
        self.ui.lineEdit_4.setEnabled(False)
        self.new_t = threading.Thread(target=start_new_thread_video, args=(self.signal, 0))
        self.new_t.start()

    def useFile(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video File", "",
                                                             "Video Files (*.avi *.mp4 *.mkv *.flv);;All Files (*)")

        if file_name:
            self.ui.lbl_isRunning.setText("运行中")
            self.ui.lineEdit.setEnabled(False)
            self.ui.lineEdit_2.setEnabled(False)
            self.ui.lineEdit_3.setEnabled(False)
            self.ui.lineEdit_4.setEnabled(False)
            self.new_t = threading.Thread(target=start_new_thread_video, args=(self.signal, file_name))
            self.new_t.start()

    def useImg(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image File", "",
                                                             "Image Files (*.jpg *.bmp *.png *.jpeg);;All Files (*)")

        if file_name:
            self.ui.lbl_isRunning.setText("运行中")
            self.ui.lineEdit.setEnabled(False)
            self.ui.lineEdit_2.setEnabled(False)
            self.ui.lineEdit_3.setEnabled(False)
            self.ui.lineEdit_4.setEnabled(False)
            self.new_t = threading.Thread(target=start_new_thread_video, args=(self.signal, file_name))
            self.new_t.start()

    def exit(self):
        sys.exit()

    def draw(self, cvimage):
        height, width, channel = cvimage.shape
        image = cvimage.data
        qImage = QtGui.QImage(image, width, height, 3 * width, QtGui.QImage.Format.Format_RGB888).rgbSwapped()
        qImage = qImage.scaledToWidth(self.width(), mode=QtCore.Qt.TransformationMode.SmoothTransformation)
        self.ui.label.setPixmap(QtGui.QPixmap.fromImage(qImage))
