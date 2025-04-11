"""
此文件用于设计用户图形交互界面
"""

import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QListWidget,
    QTextEdit,
    QStackedWidget,
    QLabel,
    QToolBar,
    QStatusBar,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon


class PackingSoftware(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能装箱软件")
        self.setMinimumSize(800, 600)

        # 设置中心区域
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 创建主布局
        self.main_layout = QVBoxLayout(self.central_widget)

        # 创建堆叠窗口用于切换不同功能界面
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        # 初始化所有界面
        self.init_ui()

        # 默认显示第一个界面
        self.stacked_widget.setCurrentIndex(0)

    def init_ui(self):
        # 创建导航栏
        self.create_navbar()

        # 创建左侧功能栏
        self.create_left_dock()

        # 创建右侧工具栏
        self.create_right_dock()

        # 创建底部日志窗口
        self.create_bottom_dock()

        # 创建状态栏
        self.statusBar().showMessage("就绪")

        # 创建所有功能页面
        self.create_pages()

    def create_navbar(self):
        # 创建菜单栏
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件")
        file_menu.addAction("新建")
        file_menu.addAction("打开")
        file_menu.addAction("保存")
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close)

        # 编辑菜单
        edit_menu = menubar.addMenu("编辑")
        edit_menu.addAction("撤销")
        edit_menu.addAction("重做")

        # 视图菜单 - 控制dock的显示/隐藏
        view_menu = menubar.addMenu("视图")
        self.toggle_left_dock_action = view_menu.addAction(
            "显示功能栏", self.toggle_left_dock
        )
        self.toggle_right_dock_action = view_menu.addAction(
            "显示工具栏", self.toggle_right_dock
        )
        self.toggle_bottom_dock_action = view_menu.addAction(
            "显示日志窗口", self.toggle_bottom_dock
        )

        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        help_menu.addAction("关于")

    def create_left_dock(self):
        # 左侧功能选择dock
        self.left_dock = QDockWidget("功能导航", self)
        self.left_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # 创建功能列表
        self.function_list = QListWidget()
        self.function_list.addItems(["条件设置", "3D结果展示", "动态仿真", "数据管理"])
        self.function_list.currentRowChanged.connect(self.switch_page)

        # 设置dock内容
        self.left_dock.setWidget(self.function_list)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)

        # 默认显示
        self.left_dock.setVisible(True)

    def create_right_dock(self):
        # 右侧工具栏dock
        self.right_dock = QDockWidget("工具栏", self)
        self.right_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # 创建工具按钮区域
        self.tool_widget = QWidget()
        self.tool_layout = QVBoxLayout(self.tool_widget)

        # 添加一些示例工具按钮
        self.condition_tools = self.create_tool_buttons(
            ["导入条件", "导出条件", "清空条件"]
        )
        self.view3d_tools = self.create_tool_buttons(["旋转", "缩放", "截图"])
        self.simulation_tools = self.create_tool_buttons(["开始", "暂停", "重置"])
        self.data_tools = self.create_tool_buttons(["导入数据", "导出数据", "分析数据"])

        # 默认隐藏所有工具组
        for tool_group in [
            self.condition_tools,
            self.view3d_tools,
            self.simulation_tools,
            self.data_tools,
        ]:
            for btn in tool_group:
                btn.setVisible(False)

        # 添加一个默认显示的按钮
        self.default_tool_btn = QPushButton("常用工具")
        self.default_tool_btn.setVisible(True)
        self.tool_layout.addWidget(self.default_tool_btn)
        self.tool_layout.addStretch()

        self.right_dock.setWidget(self.tool_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)

        # 默认显示
        self.right_dock.setVisible(True)

    def create_bottom_dock(self):
        # 底部日志窗口
        self.bottom_dock = QDockWidget("日志", self)
        self.bottom_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)

        # 创建日志文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.bottom_dock.setWidget(self.log_text)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bottom_dock)

        # 默认显示
        self.bottom_dock.setVisible(True)

    def create_pages(self):
        # 条件设置页面
        condition_page = QWidget()
        condition_layout = QVBoxLayout(condition_page)
        condition_layout.addWidget(QLabel("装箱条件设置界面"))
        self.stacked_widget.addWidget(condition_page)

        # 3D结果展示页面
        view3d_page = QWidget()
        view3d_layout = QVBoxLayout(view3d_page)
        view3d_layout.addWidget(QLabel("3D装箱结果展示界面"))
        self.stacked_widget.addWidget(view3d_page)

        # 动态仿真页面
        simulation_page = QWidget()
        simulation_layout = QVBoxLayout(simulation_page)
        simulation_layout.addWidget(QLabel("装箱过程动态仿真界面"))
        self.stacked_widget.addWidget(simulation_page)

        # 数据管理页面
        data_page = QWidget()
        data_layout = QVBoxLayout(data_page)
        data_layout.addWidget(QLabel("装箱数据管理界面"))
        self.stacked_widget.addWidget(data_page)

    def create_tool_buttons(self, tool_names):
        buttons = []
        for name in tool_names:
            btn = QPushButton(name)
            btn.setFixedHeight(30)
            self.tool_layout.insertWidget(self.tool_layout.count() - 1, btn)
            buttons.append(btn)
        return buttons

    def switch_page(self, index):
        self.stacked_widget.setCurrentIndex(index)
        self.update_tools(index)

    def update_tools(self, index):
        # 隐藏所有工具按钮
        for tool_group in [
            self.condition_tools,
            self.view3d_tools,
            self.simulation_tools,
            self.data_tools,
        ]:
            for btn in tool_group:
                btn.setVisible(False)

        # 显示当前功能对应的工具按钮
        if index == 0:  # 条件设置
            for btn in self.condition_tools:
                btn.setVisible(True)
        elif index == 1:  # 3D展示
            for btn in self.view3d_tools:
                btn.setVisible(True)
        elif index == 2:  # 动态仿真
            for btn in self.simulation_tools:
                btn.setVisible(True)
        elif index == 3:  # 数据管理
            for btn in self.data_tools:
                btn.setVisible(True)

    def toggle_left_dock(self):
        visible = self.left_dock.isVisible()
        self.left_dock.setVisible(not visible)
        self.toggle_left_dock_action.setText(
            "隐藏功能栏" if not visible else "显示功能栏"
        )

    def toggle_right_dock(self):
        visible = self.right_dock.isVisible()
        self.right_dock.setVisible(not visible)
        self.toggle_right_dock_action.setText(
            "隐藏工具栏" if not visible else "显示工具栏"
        )

    def toggle_bottom_dock(self):
        visible = self.bottom_dock.isVisible()
        self.bottom_dock.setVisible(not visible)
        self.toggle_bottom_dock_action.setText(
            "隐藏日志窗口" if not visible else "显示日志窗口"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PackingSoftware()
    window.show()
    sys.exit(app.exec_())
