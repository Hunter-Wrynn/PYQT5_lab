# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Fromwin2.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Formwin2(object):
    def setupUi(self, Formwin2):
        Formwin2.setObjectName("Formwin2")
        Formwin2.resize(571, 388)
        self.label_daichuli = QtWidgets.QLabel(Formwin2)
        self.label_daichuli.setGeometry(QtCore.QRect(80, 50, 151, 151))
        self.label_daichuli.setObjectName("label_daichuli")
        self.label_jieguo = QtWidgets.QLabel(Formwin2)
        self.label_jieguo.setGeometry(QtCore.QRect(340, 50, 151, 151))
        self.label_jieguo.setObjectName("label_jieguo")
        self.pushButton_load = QtWidgets.QPushButton(Formwin2)
        self.pushButton_load.setGeometry(QtCore.QRect(100, 220, 101, 31))
        self.pushButton_load.setObjectName("pushButton_load")
        self.pushButton_save = QtWidgets.QPushButton(Formwin2)
        self.pushButton_save.setGeometry(QtCore.QRect(350, 220, 101, 31))
        self.pushButton_save.setObjectName("pushButton_save")
        self.label_tips = QtWidgets.QLabel(Formwin2)
        self.label_tips.setGeometry(QtCore.QRect(30, 280, 51, 41))
        self.label_tips.setObjectName("label_tips")
        self.pushButton_fun0 = QtWidgets.QPushButton(Formwin2)
        self.pushButton_fun0.setGeometry(QtCore.QRect(110, 290, 101, 31))
        self.pushButton_fun0.setObjectName("pushButton_fun0")
        self.pushButton_fun1 = QtWidgets.QPushButton(Formwin2)
        self.pushButton_fun1.setGeometry(QtCore.QRect(230, 290, 101, 31))
        self.pushButton_fun1.setObjectName("pushButton_fun1")
        self.pushButton_fun11 = QtWidgets.QPushButton(Formwin2)
        self.pushButton_fun11.setGeometry(QtCore.QRect(350, 290, 101, 31))
        self.pushButton_fun11.setObjectName("pushButton_fun11")

        self.retranslateUi(Formwin2)
        QtCore.QMetaObject.connectSlotsByName(Formwin2)

    def retranslateUi(self, Formwin2):
        _translate = QtCore.QCoreApplication.translate
        Formwin2.setWindowTitle(_translate("Formwin2", "Form"))
        self.label_daichuli.setText(_translate("Formwin2", "待处理"))
        self.label_jieguo.setText(_translate("Formwin2", "结果"))
        self.pushButton_load.setText(_translate("Formwin2", "选择图片"))
        self.pushButton_save.setText(_translate("Formwin2", "保存图片"))
        self.label_tips.setText(_translate("Formwin2", "操作："))
        self.pushButton_fun0.setText(_translate("Formwin2", "水平翻转"))
        self.pushButton_fun1.setText(_translate("Formwin2", "垂直翻转"))
        self.pushButton_fun11.setText(_translate("Formwin2", "沿xy轴翻转"))
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
# ... 省略你的所有TableWidget子类的定义 ...

def main():
    # 创建Qt应用程序对象
    app = QApplication(sys.argv)

    # 创建主窗口
    MainWindow = QMainWindow()
    ui = Ui_Formwin2()
    ui.setupUi(MainWindow)

    # 显示主窗口
    MainWindow.show()

    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()