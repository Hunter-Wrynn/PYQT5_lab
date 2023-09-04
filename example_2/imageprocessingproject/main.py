import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage, qRgb

import Mainwindow
from PyQt5.QtWidgets import *
import sys

import picture_fun


class MainWindow(QWidget,Mainwindow.Ui_Mainwindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.stackedLayout = QStackedLayout(self.frame)

        self.page1 = FramePage1()
        self.page2 = FramePage2()
        self.page3 = FramePage3()
        self.page4 = FramePage4()
        self.page5 = FramePage5()
        self.page6 = FramePage6()
        self.page7 = FramePage7()

        self.stackedLayout.addWidget(self.page1)
        self.stackedLayout.addWidget(self.page2)
        self.stackedLayout.addWidget(self.page3)
        self.stackedLayout.addWidget(self.page4)
        self.stackedLayout.addWidget(self.page5)
        self.stackedLayout.addWidget(self.page6)
        self.stackedLayout.addWidget(self.page7)

        self.controller()

    def controller(self):
        self.pushButton_1.clicked.connect(self.switch1)
        self.pushButton_2.clicked.connect(self.switch2)
        self.pushButton_3.clicked.connect(self.switch3)
        self.pushButton_4.clicked.connect(self.switch4)
        self.pushButton_5.clicked.connect(self.switch5)
        self.pushButton_6.clicked.connect(self.switch6)
        self.pushButton_7.clicked.connect(self.switch7)

    def switch1(self):
        self.stackedLayout.setCurrentIndex(0)  # 索引按加入布局的顺序
    def switch2(self):
        self.stackedLayout.setCurrentIndex(1)
    def switch3(self):
        self.stackedLayout.setCurrentIndex(2)
    def switch4(self):
        self.stackedLayout.setCurrentIndex(3)
    def switch5(self):
        self.stackedLayout.setCurrentIndex(4)
    def switch6(self):
        self.stackedLayout.setCurrentIndex(5)
    def switch7(self):
        self.stackedLayout.setCurrentIndex(6)




class FramePage1(QWidget, Mainwindow.Ui_Formwin1):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_load.clicked.connect(self.openimage)
        self.pushButton_save.clicked.connect(self.saveimage)
        self.pushButton_turntoGray.clicked.connect(self.fun1)
        self.pushButton_turntotwo.clicked.connect(self.fun2)

    def openimage(self):
        fname = QFileDialog.getOpenFileName(self, '打开图片', './', "Images (*.png *.jpg *.bmp)")
        if fname[0]:
            self.label_daichuli.setPixmap(QPixmap(fname[0]))
            self.label_daichuli.setWordWrap(True)
            self.label_daichuli.setScaledContents(True)

    def saveimage(self):
        nfname = QFileDialog.getSaveFileName(self, "保存图片", "./", "Images (*.png *.jpg *.bmp)")
        if nfname[0]:
            self.label_jieguo.pixmap().save(nfname[0])

    def fun1(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        newsrc = picture_fun.gray_picture(src)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)

    def fun2(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        newsrc = picture_fun.erzhihua(src)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)


class FramePage2(QWidget, Mainwindow.Ui_Formwin2):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_load.clicked.connect(self.openimage)
        self.pushButton_save.clicked.connect(self.saveimage)
        self.pushButton_fun0.clicked.connect(self.fun1)
        self.pushButton_fun1.clicked.connect(self.fun2)
        self.pushButton_fun11.clicked.connect(self.fun3)

    def openimage(self):
        fname = QFileDialog.getOpenFileName(self, '打开图片', './', "Images (*.png *.jpg *.bmp)")
        if fname[0]:
            self.label_daichuli.setPixmap(QPixmap(fname[0]))
            self.label_daichuli.setWordWrap(True)
            self.label_daichuli.setScaledContents(True)

    def saveimage(self):
        nfname = QFileDialog.getSaveFileName(self, "保存图片", "./", "Images (*.png *.jpg *.bmp)")
        if nfname[0]:
            self.label_jieguo.pixmap().save(nfname[0])

    def fun1(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        newsrc = picture_fun.flipfun(src,0)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)

    def fun2(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        newsrc = picture_fun.flipfun(src,1)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)

    def fun3(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        newsrc = picture_fun.flipfun(src,-1)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)


class FramePage3(QWidget, Mainwindow.Ui_Formwin3):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_load.clicked.connect(self.openimage)
        self.pushButton_save.clicked.connect(self.saveimage)
        self.pushButton_lap.clicked.connect(self.lap_fun)
        self.pushButton_zhifangtu.clicked.connect(self.equalhist_fun)
        self.pushButton_gama.clicked.connect(self.gama_fun)

    def openimage(self):
        fname = QFileDialog.getOpenFileName(self, '打开图片', './', "Images (*.png *.jpg *.bmp)")
        if fname[0]:
            self.label_daichuli.setPixmap(QPixmap(fname[0]))
            self.label_daichuli.setWordWrap(True)
            self.label_daichuli.setScaledContents(True)

    def saveimage(self):
        nfname = QFileDialog.getSaveFileName(self, "保存图片", "./", "Images (*.png *.jpg *.bmp)")
        if nfname[0]:
            self.label_jieguo.pixmap().save(nfname[0])

    def lap_fun(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        newsrc = picture_fun.lap_9(src)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)

    def equalhist_fun(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        newsrc = picture_fun.cal_equalhist(src)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)

    def gama_fun(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        power=self.lineEdit_power_value.text()
        power=float(power)
        newsrc = picture_fun.gama_transfer(src,power)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)


class FramePage4(QWidget, Mainwindow.Ui_Formwin4):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_load.clicked.connect(self.openimage)
        self.pushButton_save.clicked.connect(self.saveimage)
        self.pushButton_gama.clicked.connect(self.add_noisy_fun)           #椒盐彩色
        self.pushButton_zhifangtu.clicked.connect(self.add_noise_fun)     #彩色图像添加高斯噪声

    def openimage(self):
        fname = QFileDialog.getOpenFileName(self, '打开图片', './', "Images (*.png *.jpg *.bmp)")
        if fname[0]:
            self.label_daichuli.setPixmap(QPixmap(fname[0]))
            self.label_daichuli.setWordWrap(True)
            self.label_daichuli.setScaledContents(True)

    def saveimage(self):
        nfname = QFileDialog.getSaveFileName(self, "保存图片", "./", "Images (*.png *.jpg *.bmp)")
        if nfname[0]:
            self.label_jieguo.pixmap().save(nfname[0])

    def add_noisy_fun(self):            #椒盐彩色
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        num = self.lineEdit_n_value.text()
        num = int(num)
        newsrc = picture_fun.add_noisy(src, num)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)

    def add_noise_fun(self):          #彩色图像添加高斯噪声
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        mean = self.lineEdit_mean_value.text()
        mean = int(mean)
        val=self.lineEdit_val_value.text()
        val=float(val)
        newsrc = picture_fun.add_noise(src,mean,val)
        pix = matqimage_guass(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)


class FramePage5(QWidget, Mainwindow.Ui_Formwin5):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_load.clicked.connect(self.openimage)
        self.pushButton_save.clicked.connect(self.saveimage)
        self.pushButton_fangbo.clicked.connect(self.boxFilter_fun)
        self.pushButton_zhongzhi.clicked.connect(self.medianBlur_fun)
        self.pushButton_suangbian.clicked.connect(self.bilateralFilter_fun)
        self.pushButton_gauss.clicked.connect(self.GaussianBlur_fun)
        self.pushButton_junzhi.clicked.connect(self.blur_fun)

    def openimage(self):
        fname = QFileDialog.getOpenFileName(self, '打开图片', './', "Images (*.png *.jpg *.bmp)")
        if fname[0]:
            self.label_daichuli.setPixmap(QPixmap(fname[0]))
            self.label_daichuli.setWordWrap(True)
            self.label_daichuli.setScaledContents(True)

    def saveimage(self):
        nfname = QFileDialog.getSaveFileName(self, "保存图片", "./", "Images (*.png *.jpg *.bmp)")
        if nfname[0]:
            self.label_jieguo.pixmap().save(nfname[0])

    def boxFilter_fun(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        newsrc = picture_fun.boxFilterfun(src)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)

    def medianBlur_fun(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        newsrc = picture_fun.medianBlurfun(src)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)

    def bilateralFilter_fun(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        newsrc = picture_fun.bilateralFilterfun(src)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)

    def GaussianBlur_fun(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        newsrc = picture_fun.GaussianBlurfun(src)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)

    def blur_fun(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        newsrc = picture_fun.blurfun(src)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)


class FramePage6(QWidget, Mainwindow.Ui_Formwin6):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_load.clicked.connect(self.openimage)
        self.pushButton_save.clicked.connect(self.saveimage)
        self.pushButton_sift.clicked.connect(self.siftfun)
        self.pushButton_lunkuo.clicked.connect(self.lunkuofun)

    def openimage(self):
        fname = QFileDialog.getOpenFileName(self, '打开图片', './', "Images (*.png *.jpg *.bmp)")
        if fname[0]:
            self.label_daichuli.setPixmap(QPixmap(fname[0]))
            self.label_daichuli.setWordWrap(True)
            self.label_daichuli.setScaledContents(True)

    def saveimage(self):
        nfname = QFileDialog.getSaveFileName(self, "保存图片", "./", "Images (*.png *.jpg *.bmp)")
        if nfname[0]:
            self.label_jieguo.pixmap().save(nfname[0])


    def siftfun(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        newsrc = picture_fun.sift_fun(src)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)

    def lunkuofun(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        newsrc = picture_fun.morphologyExfun(src)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)


class FramePage7(QWidget, Mainwindow.Ui_Formwin7):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_load.clicked.connect(self.openimage)
        self.pushButton_save.clicked.connect(self.saveimage)
        self.pushButton_xiufu.clicked.connect(self.xiufu_fun)

    def openimage(self):
        fname = QFileDialog.getOpenFileName(self, '打开图片', './', "Images (*.png *.jpg *.bmp)")
        if fname[0]:
            self.label_daichuli.setPixmap(QPixmap(fname[0]))
            self.label_daichuli.setWordWrap(True)
            self.label_daichuli.setScaledContents(True)

    def saveimage(self):
        nfname = QFileDialog.getSaveFileName(self, "保存图片", "./", "Images (*.png *.jpg *.bmp)")
        if nfname[0]:
            self.label_jieguo.pixmap().save(nfname[0])

    def xiufu_fun(self):
        qimg = self.label_daichuli.pixmap()
        src = qimage2mat(qimg)
        newsrc = picture_fun.xiufu(src)
        pix = matqimage(newsrc)
        self.label_jieguo.setPixmap(pix)
        self.label_jieguo.setWordWrap(True)
        self.label_jieguo.setScaledContents(True)
#############################################################################################################

def qimage2mat(qtpixmap):    #qtpixmap转opencv
    qimg = qtpixmap.toImage()
    temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
    temp_shape += (4,)
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
    result = result[..., :3]
    return result


def matqimage(cvimg):       #opencv转QImage
    if cvimg.ndim==2:              #单通道
        height, width= cvimg.shape
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        cvimg = QImage(cvimg.data, width, height, QImage.Format_RGB888)
        pix = QPixmap.fromImage(cvimg)
        return pix
    else:                          #多个通道
        width = cvimg.shape[1]
        height = cvimg.shape[0]
        pixmap = QPixmap(width, height)  # 根据已知的高度和宽度新建一个空的QPixmap,
        qimg = pixmap.toImage()         # 将pximap转换为QImage类型的qimg
        for row in range(0, height):
            for col in range(0, width):
                b = cvimg[row, col, 0]
                g = cvimg[row, col, 1]
                r = cvimg[row, col, 2]
                pix = qRgb(r, g, b)
                qimg.setPixel(col, row, pix)
                pix = QPixmap.fromImage(qimg)
        return pix

def matqimage_guass(cvimg):          #opencv转QImage   #guass 专用####
    if cvimg.ndim==2:                    #单通道
        height, width= cvimg.shape
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        cvimg = QImage(cvimg.data, width, height, QImage.Format_RGB888)
        pix = QPixmap.fromImage(cvimg)
        return pix
    else:                                #多个通道
        width = cvimg.shape[1]  # 获取图片宽度
        height = cvimg.shape[0]  # 获取图片高度
        pixmap = QPixmap(width, height)  # 根据已知的高度和宽度新建一个空的QPixmap,
        qimg = pixmap.toImage()  # 将pximap转换为QImage类型的qimg
        for row in range(0, height):
            for col in range(0, width):
                b = int(cvimg[row, col, 0]*255)            #高斯加噪归一化了 要*255
                g = int(cvimg[row, col, 1]*255)
                r = int(cvimg[row, col, 2]*255)
                pix = qRgb(r, g, b)
                qimg.setPixel(col, row, pix)
                pix = QPixmap.fromImage(qimg)
        return pix  # 转换完成，返回

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())