"""
此文件用于设计用户图形交互界面
"""

import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from src.interface.qt_desgin.ui_main_window import Ui_PackingSoftware


class PackingSoftware(QMainWindow, Ui_PackingSoftware):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 初始化界面
        self.init_ui()

        # 连接信号槽
        self.connect_signals()

        # 初始状态
        self.switch_page(0)

    def init_ui(self):
        # 组织工具按钮组
        self.tool_groups = {
            0: [
                self.import_condition_btn,
                self.export_condition_btn,
                self.clear_condition_btn,
            ],
            1: [],  # 3D展示工具按钮(需要在UI中添加)
            2: [],  # 动态仿真工具按钮(需要在UI中添加)
            3: [],  # 数据管理工具按钮(需要在UI中添加)
        }

        # 设置默认工具按钮可见性
        self.default_tool_btn.setVisible(True)

        # 设置窗口可自由调整大小
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMaximizeButtonHint)

    def connect_signals(self):
        # 连接功能列表切换信号
        self.function_list.currentRowChanged.connect(self.switch_page)

        # 连接视图菜单动作
        self.action_toggle_left_dock.triggered.connect(self.toggle_left_dock)
        self.action_toggle_right_dock.triggered.connect(self.toggle_right_dock)
        self.action_toggle_bottom_dock.triggered.connect(self.toggle_bottom_dock)

        # 连接文件菜单动作
        self.action_exit.triggered.connect(self.close)

    def switch_page(self, index):
        """切换功能页面"""
        self.stacked_widget.setCurrentIndex(index)
        self.update_tools(index)

    def update_tools(self, index):
        """更新工具栏按钮显示"""
        # 隐藏所有工具按钮
        for group in self.tool_groups.values():
            for btn in group:
                btn.setVisible(False)

        # 显示当前功能对应的工具按钮
        for btn in self.tool_groups.get(index, []):
            btn.setVisible(True)

        # 如果没有工具按钮，显示默认按钮
        if not self.tool_groups.get(index, []):
            self.default_tool_btn.setVisible(True)
        else:
            self.default_tool_btn.setVisible(False)

    def toggle_left_dock(self):
        """切换左侧功能栏显示"""
        visible = not self.left_dock.isVisible()
        self.left_dock.setVisible(visible)
        self.action_toggle_left_dock.setText("隐藏功能栏" if visible else "显示功能栏")

    def toggle_right_dock(self):
        """切换右侧工具栏显示"""
        visible = not self.right_dock.isVisible()
        self.right_dock.setVisible(visible)
        self.action_toggle_right_dock.setText("隐藏工具栏" if visible else "显示工具栏")

        # 当工具栏关闭时，只显示默认按钮
        if not visible:
            for group in self.tool_groups.values():
                for btn in group:
                    btn.setVisible(False)
            self.default_tool_btn.setVisible(True)

    def toggle_bottom_dock(self):
        """切换底部日志窗口显示"""
        visible = not self.bottom_dock.isVisible()
        self.bottom_dock.setVisible(visible)
        self.action_toggle_bottom_dock.setText(
            "隐藏日志窗口" if visible else "显示日志窗口"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用程序样式(可选)
    # app.setStyle('Fusion')

    window = PackingSoftware()
    window.show()
    sys.exit(app.exec_())
