import datetime
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QTimer, QFile, QTextStream
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QMessageBox, QFileDialog
from PyQt5.uic import loadUi

from unet_model import *
from unet_train import *


class UnetDialog(QDialog):
    def __init__(self):
        super(UnetDialog, self).__init__()

        self.model = None

        self.SetUI()
        self.SetConnect()

    def SetUI(self):
        loadUi('main.ui', self)
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setFixedSize(self.size())

    def SetConnect(self):
        self.btn_load_img.clicked.connect(self.btn_load_img_clicked)
        self.btn_start.clicked.connect(self.btn_start_clicked)

    def btn_load_img_clicked(self):
        pass

    def btn_start_clicked(self):
        pass

    def keyPressEvent(self, event):
        if not event.key() == Qt.Key_Escape:
            super(UnetDialog, self).keyPressEvent(event)


app = QApplication(sys.argv)
main = UnetDialog()
main.show()
sys.exit(app.exec_())
