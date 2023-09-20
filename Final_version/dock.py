from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsView, QPushButton
from PyQt5.QtCore import Qt

class DraggableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.last_mouse_pos = None
        self.is_draggable = False  # 初始化为不可拖动

    def setDraggable(self, draggable):
        self.is_draggable = draggable

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_draggable:
            self.last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_draggable:
            self.last_mouse_pos = None

    def mouseMoveEvent(self, event):
        if self.is_draggable and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.setGeometry(self.geometry().translated(delta.x(), delta.y()))
            self.last_mouse_pos = event.pos()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = DraggableGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(310, 170, 256, 192))
        self.graphicsView.setObjectName("graphicsView")

        self.pushButtonMove = QPushButton(self.centralwidget)
        self.pushButtonMove.setGeometry(QtCore.QRect(50, 50, 100, 30))
        self.pushButtonMove.setText("Enable/Disable Move")
        self.pushButtonMove.clicked.connect(self.toggleMove)  # 连接按钮的点击事件

        self.pushButtonZoomIn = QPushButton(self.centralwidget)
        self.pushButtonZoomIn.setGeometry(QtCore.QRect(50, 100, 100, 30))
        self.pushButtonZoomIn.setText("Zoom In")
        self.pushButtonZoomIn.clicked.connect(self.zoomIn)  # 连接按钮的点击事件

        MainWindow.setCentralWidget(self.centralwidget)

    def toggleMove(self):
        # 切换是否允许移动
        self.graphicsView.setDraggable(not self.graphicsView.is_draggable)

    def zoomIn(self):
        # 放大 GraphicsView 的尺寸
        current_geometry = self.graphicsView.geometry()
        new_geometry = current_geometry.adjusted(-10, -10, 10, 10)  # 增加一些尺寸
        self.graphicsView.setGeometry(new_geometry)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


import cv2


