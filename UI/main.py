import sys
import os

# 添加 ultralytics 目录的父目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import PyQt6.QtWidgets as QtWidgets
from qt_material import apply_stylesheet

import mainwindow


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mw = mainwindow.MainWindow()

    apply_stylesheet(app,theme='dark_teal.xml')

    mw.show()
    sys.exit(app.exec())

