# -*- coding: utf-8 -*-  #
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("关于")
        self.resize(100, 30)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint
        )  # 去掉问号按钮
        layout = QVBoxLayout()

        font = self.font()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(8)
        self.setFont(font)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("智能装箱软件 v1.0"))
        layout.addWidget(QLabel("版权所有 © 2025"))
        self.setLayout(layout)
