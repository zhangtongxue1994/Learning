"""
------------------------------------------------
File Name: main_GUI.py
Description:
Author: zhangtongxue
Date: 2019/10/26 10:58
-------------------------------------------------
"""
import sys
import resource
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QCoreApplication


# 继承自QWidget
class Calculater(QWidget):

    def __init__(self):
        super().__init__()

        # 使用initUI()方法创建一个GUI
        self.initUI()

    def initUI(self):
        """
        GUI的初始化函数
        """
        # 把窗口放到屏幕上并且设置窗口大小
        self.center()  # 窗口居中
        self.resize(500, 400)
        self.setWindowTitle('GUI Demo')  # 添加标题
        self.setWindowIcon(QIcon('icon.png'))  # 添加图标

        QToolTip.setFont(QFont('SansSerif', 10))  # 设置字体字号
        Font = QFont('SansSerif', 18)

        # 按钮的栅格布局
        grid = QGridLayout()
        self.setLayout(grid)

        names = ['Cls', 'Bck', '', 'Close',
                 '7', '8', '9', '/',
                 '4', '5', '6', '*',
                 '1', '2', '3', '-',
                 '0', '.', '=', '+']

        positions = [(i, j) for i in range(5) for j in range(4)]

        for position, name in zip(positions, names):

            if name == '':
                continue
            button = QPushButton(name)
            grid.addWidget(button, *position)

        self.show()

    def closeEvent(self, event):
        """
        如果关闭QWidget，就会产生一个QCloseEvent，并且把它传入到closeEvent函数的event参数中
        """
        reply = QMessageBox.question(self, '提示信息',
                                     "确定要退出程序吗?",
                                     QMessageBox.Yes |
                                     QMessageBox.No,
                                     QMessageBox.Yes)  # 最后一个参数是默认按钮

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def center(self):
        """
        设置GUI窗口居中
        """
        qr = self.frameGeometry()  # 获得主窗口所在的框架
        cp = QDesktopWidget().availableGeometry().center()  # 获取显示器的分辨率，然后得到屏幕中间点的位置
        qr.moveCenter(cp)  # 主窗口框架的中心点放置到屏幕的中心位置
        self.move(qr.topLeft())  # 通过move函数把主窗口的左上角移动到其框架的左上角，这样就把窗口居中了。


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Calculater()
    sys.exit(app.exec_())
