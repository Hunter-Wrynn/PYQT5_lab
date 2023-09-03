import cv2
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class FileSystemTreeView(QTreeView, QDockWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.mainwindow = parent
        self.fileSystemModel = QFileSystemModel()
        self.fileSystemModel.setRootPath('.')
        self.setModel(self.fileSystemModel)
        # 隐藏size,date等列
        self.setColumnWidth(0, 200)
        self.setColumnHidden(1, True)
        self.setColumnHidden(2, True)
        self.setColumnHidden(3, True)
        # 不显示标题栏
        self.header().hide()
        # 设置动画
        self.setAnimated(True)
        # 选中不显示虚线
        self.setFocusPolicy(Qt.NoFocus)
        self.doubleClicked.connect(self.select_image)
        self.setMinimumWidth(200)

    def select_image(self, file_index):
        file_name = self.fileSystemModel.filePath(file_index)
        if file_name.endswith(('.jpg', '.png', '.bmp')):
            src_img = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), -1)
            self.mainwindow.change_image(src_img)

from PyQt5.QtWidgets import QApplication, QMainWindow

# ... 省略你的所有TableWidget子类的定义 ...

def main():
    app = QApplication([])
    main_window = QMainWindow()
    main_window.setWindowTitle("My PyQt Application")
    main_window.setGeometry(100, 100, 800, 600)

    # 创建一个TableWidget子类的实例，例如 GrayingTableWidget
    table_widget = FileSystemTreeView(main_window)

    # 将table_widget添加到主窗口的布局中

    main_window.show()
    app.exec_()

if __name__ == "__main__":
    main()