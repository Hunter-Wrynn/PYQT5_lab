# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets

import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage, qRgb
import algorithm as ag
import time
import autopy
import HandTrackingModule as ht
from PyQt5.QtWidgets import *
import sys


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        # 设置窗口背景颜色
        MainWindow.setStyleSheet("background-color: #545454;")
        self.centralwidget.setStyleSheet("background-color: #2C2C2C;")


        self.label_daichuli = QtWidgets.QLabel(self.centralwidget)
        self.label_daichuli.setGeometry(QtCore.QRect(100, 100, 300, 300))
        self.label_daichuli.setObjectName("label_daichuli")
        self.label_jieguo = QtWidgets.QLabel(self.centralwidget)
        self.label_jieguo.setGeometry(QtCore.QRect(500, 100, 300, 300))
        self.label_jieguo.setObjectName("label_jieguo")

        self.label_daichuli.setStyleSheet("background-color: #FFFFFF;")
        self.label_jieguo.setStyleSheet("background-color: #FFFFFF;")

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        self.menu_1 = QtWidgets.QMenu(self.menubar)
        self.menu_1.setObjectName("menu_1")
        self.menubar.addAction(self.menu_1.menuAction())
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menubar.addAction(self.menu_2.menuAction())
        MainWindow.setMenuBar(self.menubar)

        self.actionopen = QtWidgets.QAction(MainWindow)
        self.actionopen.setObjectName("actionopen")
        self.menu_1.addAction(self.actionopen)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # 自定义菜单项样式
        menubar_style = "QMenuBar::item { color: #FFFFFF; }" \
                        "QMenuBar::selected{background-color:#FFFFFF;}"  # 设置菜单项文字颜色为
        self.menubar.setStyleSheet(menubar_style)

        menu_style="QMenu {background-color:#FFFFFF;border:1px solid rgba(82,130,164,1);}"

        self.menu_1.setStyleSheet(menu_style)

        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolBar)
        self.action_tool_1 = QtWidgets.QAction(MainWindow)
        self.action_tool_1.setObjectName("action_tool_1")
        self.action_tool_2 = QtWidgets.QAction(MainWindow)
        self.action_tool_2.setObjectName("action_tool_2")
        self.action_tool_3 = QtWidgets.QAction(MainWindow)
        self.action_tool_3.setObjectName("action_tool_3")


        #信号槽
        self.toolBar.addAction(self.action_tool_1)
        self.toolBar.addAction(self.action_tool_2)
        self.toolBar.addAction(self.action_tool_3)
        self.actionopen.triggered.connect(self.open_img)
        self.action_tool_2.triggered.connect(self.flip_img)
        self.action_tool_1.triggered.connect(self.gray_img)
        self.action_tool_3.triggered.connect(self.VR)



        self.retranslateUi(MainWindow)
        self.processed_image=None


        QtCore.QMetaObject.connectSlotsByName(MainWindow)





    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_daichuli.setText(_translate("MainWindow", "待处理"))
        self.label_jieguo.setText(_translate("MainWindow", "结果"))
        self.menu_1.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "编辑"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.action_tool_1.setText(_translate("MainWindow", "灰度"))
        self.action_tool_2.setText(_translate("MainWindow", "平滑"))
        self.action_tool_3.setText(_translate("MainWindow", "VR"))
        self.actionopen.setText(_translate("MainWindow", "打开"))


    def open_img(self):

        fname = QFileDialog.getOpenFileName(None, '打开图片', './', "Images (*.png *.jpg *.bmp)")
        if fname[0]:

            self.label_daichuli.setPixmap(QPixmap(fname[0]))
            self.label_daichuli.setWordWrap(True)
            self.label_daichuli.setScaledContents(True)

    def save_img(self):
        nfname = QFileDialog.getSaveFileName(self, "保存图片", "./", "Images (*.png *.jpg *.bmp)")
        if nfname[0]:
            self.label_jieguo.pixmap().save(nfname[0])

    def gray_img(self):
        if self.processed_image is not None:
            if len(self.processed_image.shape) == 2:
                return
            else:
                newsrc = ag.gray_picture(self.processed_image)
                pix = ag.matqimage(newsrc)
                self.label_jieguo.setPixmap(pix)
                self.label_jieguo.setWordWrap(True)
                self.label_jieguo.setScaledContents(True)
        else:
            qimg = self.label_daichuli.pixmap()
            src = ag.qimage2mat(qimg)
            newsrc = ag.gray_picture(src)
            pix = ag.matqimage(newsrc)
            self.label_jieguo.setPixmap(pix)
            self.label_jieguo.setWordWrap(True)
            self.label_jieguo.setScaledContents(True)
        # 存储灰度化后的图像数据
        self.processed_image = newsrc


    def flip_img(self):
        if self.processed_image is not None:
            newsrc = ag.flip_picture(self.processed_image, 0)
            pix = ag.matqimage(newsrc)
            self.label_jieguo.setPixmap(pix)
            self.label_jieguo.setWordWrap(True)
            self.label_jieguo.setScaledContents(True)
        else:
            qimg = self.label_daichuli.pixmap()
            src = ag.qimage2mat(qimg)
            newsrc = ag.flip_picture(src,0)
            pix = ag.matqimage(newsrc)
            self.label_jieguo.setPixmap(pix)
            self.label_jieguo.setWordWrap(True)
            self.label_jieguo.setScaledContents(True)

        self.processed_image = newsrc


    def VR(self):
        pTime = 0
        width, height = 640, 480
        frameR = 100
        smoothening = 8
        prev_x, prev_y = 0, 0
        curr_x, curr_y = 0, 0

        cap = cv2.VideoCapture(0)
        cap.set(3, width)
        cap.set(4, height)

        detector = ht.handDetector(maxHands=1)
        screen_width, screen_height = autopy.screen.size()

        while True:
            success, img = cap.read()
            img = detector.findHands(img)
            lmlist, bbox = detector.findPosition(img)

            if len(lmlist) != 0:
                x1, y1 = lmlist[8][1:]
                x2, y2 = lmlist[12][1:]

                fingers = detector.fingersUp()
                cv2.rectangle(img, (frameR, frameR), (width - frameR, height - frameR), (255, 0, 255), 2)

                if fingers[1] == 1 and fingers[2] == 0:
                    x3 = np.interp(x1, (frameR, width - frameR), (0, screen_width))
                    y3 = np.interp(y1, (frameR, height - frameR), (0, screen_height))

                    curr_x = prev_x + (x3 - prev_x) / smoothening
                    curr_y = prev_y + (y3 - prev_y) / smoothening

                    autopy.mouse.move(screen_width - curr_x, curr_y)
                    cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
                    prev_x, prev_y = curr_x, curr_y

                if fingers[1] == 1 and fingers[2] == 1:
                    length, img, lineInfo = detector.findDistance(8, 12, img)

                    if length < 40:
                        cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                        autopy.mouse.click()

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.imshow("Image", img)
            cv2.waitKey(1)

from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
# ... 省略你的所有TableWidget子类的定义 ...
def main():
    # 创建Qt应用程序对象
    app = QApplication(sys.argv)

    # 创建主窗口
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()

    ui.setupUi(MainWindow)

    # 显示主窗口
    MainWindow.show()

    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

from PyQt5 import QtCore, QtGui, QtDesigner


def dump_ui(widget, path):
    builder = QtDesigner.QFormBuilder()
    stream = QtCore.QFile(path)
    stream.open(QtCore.QIODevice.WriteOnly)
    builder.save(stream, widget)
    stream.close()


app = QtGui.QApplication([''])

dialog = QtGui.QDialog()
Ui_Dialog().setupUi(dialog)

dialog.show()

dump_ui(dialog, 'myui.ui')