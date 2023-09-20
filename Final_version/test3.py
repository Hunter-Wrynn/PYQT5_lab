from PyQt5.QtWidgets import QApplication, QMainWindow
from test1 import Ui_MainWindow as UM
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog
import sys
# 在这个文件中的代码中使用 Ui_MainWindow 类
# This Python file uses the following encoding: utf-8
import sys
from pathlib import Path

from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QQmlContext

KMCUI_PATH = Path(__file__).resolve().parent / "3rdparty/KmcUI"
QML_PATH = Path(__file__).resolve().parent / "qml"
sys.path.append(str(KMCUI_PATH))
import cvtools
from resources import *
from PyKmc.models import TreeModel, TreeNode
from python.image_loader import *
from python.bridge import *

print( Path(__file__).resolve().parent)
def main2():
    app = QGuiApplication(sys.argv)
    QGuiApplication.setOrganizationName("seu")
    QGuiApplication.setOrganizationDomain("seu.cvtools")

    imageProvider = ImageProvider(ImageProvider.ProviderType.Image)
    engine = QQmlApplicationEngine()
    engine.addImageProvider(imageProvider.providerId(), imageProvider)
    engine.rootContext().setContextProperty("imageProvider",imageProvider)
    qml_file = QML_PATH / "MainForm.qml"
    engine.addImportPath(KMCUI_PATH / "src/imports")
    engine.load(qml_file)
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.button = QtWidgets.QPushButton("Show UI", self)  # 创建按钮
        self.button.setGeometry(10,10,50,20)
        self.button_2 = QtWidgets.QPushButton("Show UI_2", self)  # 创建按钮
        self.button_2.setGeometry(10, 40, 50, 20)
        self.button.clicked.connect(main2)  # 将按钮的点击事件连接到方法

        self.button_2.clicked.connect(self.show_ui_main)
    def show_ui_main(self):
        self.ui_main=UiMainWindow()
        self.ui_main.show()


class UiMainWindow(QMainWindow,UM):
    def __init__(self):
        super(UiMainWindow,self).__init__()
        self.setupUi(self)



def main():
    app = QApplication(sys.argv)  # 创建应用程序对象
    app.setStyleSheet(open('./styleSheet.qss', encoding='utf-8').read())
    # 创建主窗口
    main_window = MyWindow()
    icon = QtGui.QIcon("picture/ICON.jpg")  # 将 "path_to_icon.png" 替换为你的图标文件路径
    main_window.setWindowIcon(icon)
    main_window.show()  # 显示主窗口

    sys.exit(app.exec_())  # 运行应用程序的主事件循环

if __name__ == "__main__":
    main()  # 调用 main 函数启动应用程序