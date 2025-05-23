# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets

from interface.gui.widgets.simulation_view import SimulationView


class Ui_PackingSoftware(object):
    def setupUi(self, PackingSoftware):
        PackingSoftware.setObjectName("PackingSoftware")
        PackingSoftware.resize(800, 600)
        PackingSoftware.setMinimumSize(QtCore.QSize(800, 600))

        # 1. 创建中央部件和主布局
        self.centralwidget = QtWidgets.QWidget(PackingSoftware)
        self.centralwidget.setObjectName("centralwidget")
        self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_layout.setObjectName("main_layout")

        # 2. 创建堆叠窗口部件和各页面
        self.stacked_widget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stacked_widget.setObjectName("stacked_widget")

        # -------------------------------------------- 条件界面 ----------------------------------------------------
        self.page_condition = QtWidgets.QWidget()
        self.page_condition.setObjectName("page_condition")
        self.condition_layout = QtWidgets.QVBoxLayout(self.page_condition)
        self.condition_layout.setObjectName("condition_layout")

        # 创建滚动区域
        self.scroll_area = QtWidgets.QScrollArea(self.page_condition)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("scroll_area")
        self.scroll_content = QtWidgets.QWidget()
        self.scroll_content.setObjectName("scroll_content")
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_content)
        self.scroll_layout.setObjectName("scroll_layout")

        # 条件设置表单
        self.condition_form = QtWidgets.QGroupBox("装箱条件设置")
        self.condition_form_layout = QtWidgets.QFormLayout(self.condition_form)

        # 集装箱选择
        self.container_combo = QtWidgets.QComboBox()
        self.container_combo.setObjectName("container_combo")
        self.condition_form_layout.addRow("集装箱类型:", self.container_combo)

        # 托盘使用选择
        self.pallet_check = QtWidgets.QCheckBox("使用托盘")
        self.pallet_check.setObjectName("pallet_check")
        self.condition_form_layout.addRow(self.pallet_check)

        # 托盘类型选择
        self.pallet_combo = QtWidgets.QComboBox()
        self.pallet_combo.setObjectName("pallet_combo")
        self.pallet_combo.setEnabled(False)
        self.condition_form_layout.addRow("托盘类型:", self.pallet_combo)

        # 运输方式选择
        self.transport_combo = QtWidgets.QComboBox()
        self.transport_combo.addItems(["海运", "空运"])
        self.transport_combo.setObjectName("transport_combo")
        self.condition_form_layout.addRow("运输方式:", self.transport_combo)

        # 货物管理区域
        self.product_group = QtWidgets.QGroupBox("待装箱货物")
        self.product_layout = QtWidgets.QVBoxLayout(self.product_group)

        # 货物表格
        self.product_table = QtWidgets.QTableWidget()
        self.product_table.setColumnCount(5)
        self.product_table.setHorizontalHeaderLabels(
            ["SKU", "名称", "数量", "尺寸(mm)", "重量(kg)"]
        )
        self.product_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.product_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.product_layout.addWidget(self.product_table)

        self.scroll_layout.addWidget(self.condition_form)
        self.scroll_layout.addWidget(self.product_group)
        self.scroll_area.setWidget(self.scroll_content)
        self.condition_layout.addWidget(self.scroll_area)

        self.stacked_widget.addWidget(self.page_condition)

        # -------------------------------------------- 3D界面 ----------------------------------------------------
        self.page_3dview = QtWidgets.QWidget()
        self.page_3dview.setObjectName("page_3dview")
        self.view3d_layout = QtWidgets.QVBoxLayout(self.page_3dview)
        self.view3d_layout.setObjectName("view3d_layout")

        # 添加3D视图容器和控制按钮
        self.view3d_container = QtWidgets.QWidget()
        self.view3d_container.setObjectName("view3d_container")
        self.view3d_layout.addWidget(
            self.view3d_container, 1
        )  # 主视图区域占据大部分空间

        self.stacked_widget.addWidget(self.page_3dview)

        # -------------------------------------------- 仿真界面 ----------------------------------------------------
        self.page_simulation = SimulationView()
        self.page_simulation.setObjectName("page_simulation")
        self.page_simulation.setLayout(QtWidgets.QVBoxLayout())  # 空布局
        self.stacked_widget.addWidget(self.page_simulation)

        # -------------------------------------------- 报表界面 ----------------------------------------------------
        self.page_statement = QtWidgets.QWidget()
        self.page_statement.setObjectName("page_statement")
        self.statement_layout = QtWidgets.QVBoxLayout(self.page_statement)
        self.statement_layout.setObjectName("statement_layout")
        self.label_statement = QtWidgets.QLabel(self.page_statement)
        self.label_statement.setObjectName("label_statement")
        self.statement_layout.addWidget(self.label_statement)

        self.stacked_widget.addWidget(self.page_statement)

        # -------------------------------------------- 数据界面 ----------------------------------------------------
        self.page_data = QtWidgets.QWidget()
        self.page_data.setObjectName("page_data")
        self.data_layout = QtWidgets.QVBoxLayout(self.page_data)
        self.data_layout.setContentsMargins(0, 0, 0, 0)  # 去掉边距
        self.data_layout.setSpacing(0)  # 去掉间距

        self.data_switcher = QtWidgets.QWidget(self.page_data)
        self.data_switcher.setObjectName("data_switcher")
        self.switcher_layout = QtWidgets.QHBoxLayout(self.data_switcher)
        self.switcher_layout.setObjectName("switcher_layout")
        # 数据切换按钮样式
        self.data_switcher.setStyleSheet(
            """
            QWidget#data_switcher {
                border-bottom: 1px solid #ccc;
                padding: 5px;
            }
            QPushButton {
                min-height: 25px;
                margin: 0 2px;
            }
        """
        )

        self.btn_product = QtWidgets.QPushButton(self.data_switcher)
        self.btn_product.setObjectName("btn_product")
        self.switcher_layout.addWidget(self.btn_product)

        self.btn_pallet = QtWidgets.QPushButton(self.data_switcher)
        self.btn_pallet.setObjectName("btn_pallet")
        self.switcher_layout.addWidget(self.btn_pallet)

        self.btn_container = QtWidgets.QPushButton(self.data_switcher)
        self.btn_container.setObjectName("btn_container")
        self.switcher_layout.addWidget(self.btn_container)

        self.data_layout.addWidget(self.data_switcher)

        self.data_table = QtWidgets.QTableWidget(self.page_data)
        self.data_table.setObjectName("data_table")
        self.data_layout.addWidget(self.data_table, 1)

        self.stacked_widget.addWidget(self.page_data)
        self.main_layout.addWidget(self.stacked_widget)

        PackingSoftware.setCentralWidget(self.centralwidget)

        # 3. 创建菜单栏和菜单项（必须在设置中央部件之后）
        self.menubar = QtWidgets.QMenuBar(PackingSoftware)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")

        # 先创建所有动作(Actions)
        self.action_new = QtWidgets.QAction(PackingSoftware)
        self.action_new.setObjectName("action_new")

        self.action_open = QtWidgets.QAction(PackingSoftware)
        self.action_open.setObjectName("action_open")

        self.action_save = QtWidgets.QAction(PackingSoftware)
        self.action_save.setObjectName("action_save")

        self.action_exit = QtWidgets.QAction(PackingSoftware)
        self.action_exit.setObjectName("action_exit")

        self.action_undo = QtWidgets.QAction(PackingSoftware)
        self.action_undo.setObjectName("action_undo")

        self.action_redo = QtWidgets.QAction(PackingSoftware)
        self.action_redo.setObjectName("action_redo")

        self.action_toggle_left_dock = QtWidgets.QAction(PackingSoftware)
        self.action_toggle_left_dock.setObjectName("action_toggle_left_dock")

        self.action_toggle_right_dock = QtWidgets.QAction(PackingSoftware)
        self.action_toggle_right_dock.setObjectName("action_toggle_right_dock")

        self.action_toggle_bottom_dock = QtWidgets.QAction(PackingSoftware)
        self.action_toggle_bottom_dock.setObjectName("action_toggle_bottom_dock")

        self.action_about = QtWidgets.QAction(PackingSoftware)
        self.action_about.setObjectName("action_about")

        self.action_settings = QtWidgets.QAction(PackingSoftware)
        self.action_settings.setObjectName("action_settings")

        self.theme_settings = QtWidgets.QAction(PackingSoftware)
        self.theme_settings.setObjectName("theme_settings")

        self.language_settings = QtWidgets.QAction(PackingSoftware)
        self.language_settings.setObjectName("language_settings")

        # 创建菜单
        self.menu_file = QtWidgets.QMenu(self.menubar)
        self.menu_file.setObjectName("menu_file")

        self.menu_edit = QtWidgets.QMenu(self.menubar)
        self.menu_edit.setObjectName("menu_edit")

        self.menu_view = QtWidgets.QMenu(self.menubar)
        self.menu_view.setObjectName("menu_view")

        self.menu_help = QtWidgets.QMenu(self.menubar)
        self.menu_help.setObjectName("menu_help")

        self.menu_settings = QtWidgets.QMenu(self.menubar)
        self.menu_settings.setObjectName("menu_settings")

        # 添加菜单项
        self.menu_file.addAction(self.action_new)
        self.menu_file.addAction(self.action_open)
        self.menu_file.addAction(self.action_save)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.action_exit)

        self.menu_edit.addAction(self.action_undo)
        self.menu_edit.addAction(self.action_redo)

        self.menu_view.addAction(self.action_toggle_left_dock)
        self.menu_view.addAction(self.action_toggle_right_dock)
        self.menu_view.addAction(self.action_toggle_bottom_dock)

        self.menu_help.addAction(self.action_about)

        self.menu_settings.addAction(self.action_settings)
        self.menu_settings.addAction(self.theme_settings)
        self.menu_settings.addAction(self.language_settings)

        # 添加菜单到菜单栏
        self.menubar.addAction(self.menu_file.menuAction())
        self.menubar.addAction(self.menu_edit.menuAction())
        self.menubar.addAction(self.menu_view.menuAction())
        self.menubar.addAction(self.menu_settings.menuAction())
        self.menubar.addAction(self.menu_help.menuAction())

        PackingSoftware.setMenuBar(self.menubar)

        # 4. 创建状态栏
        self.statusbar = QtWidgets.QStatusBar(PackingSoftware)
        self.statusbar.setObjectName("statusbar")
        PackingSoftware.setStatusBar(self.statusbar)

        # 5. 创建停靠窗口
        # 左侧停靠窗口 - 功能导航
        self.left_dock = QtWidgets.QDockWidget(PackingSoftware)
        self.left_dock.setObjectName("left_dock")
        self.left_dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )

        self.function_list = QtWidgets.QListWidget()
        self.function_list.setObjectName("function_list")

        # 添加功能列表项
        items = ["条件设置", "结果展示", "动态仿真", "报表打印", "数据管理"]
        for item in items:
            self.function_list.addItem(item)

        self.left_dock.setWidget(self.function_list)
        PackingSoftware.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.left_dock)

        # 右侧停靠窗口 - 工具栏
        self.right_dock = QtWidgets.QDockWidget(PackingSoftware)
        self.right_dock.setObjectName("right_dock")
        self.right_dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )

        self.tool_widget = QtWidgets.QWidget()
        self.tool_widget.setObjectName("tool_widget")
        self.tool_layout = QtWidgets.QVBoxLayout(self.tool_widget)
        self.tool_layout.setContentsMargins(5, 5, 5, 5)  # 设置边距
        self.tool_layout.setSpacing(5)  # 设置间距
        self.tool_layout.setAlignment(QtCore.Qt.AlignTop)  # 设置布局对齐方式（靠上）
        self.tool_layout.setObjectName("tool_layout")

        # 垂直间距
        self.verticalSpacer = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )

        # 添加工具按钮
        self.clear_condition_btn = QtWidgets.QPushButton(self.tool_widget)
        self.clear_condition_btn.setMinimumSize(QtCore.QSize(0, 30))
        self.clear_condition_btn.setObjectName("clear_condition_btn")
        self.clear_condition_btn.setVisible(True)  # 改为默认可见

        self.generate_solution_btn = QtWidgets.QPushButton(self.tool_widget)
        self.generate_solution_btn.setMinimumSize(QtCore.QSize(0, 30))
        self.generate_solution_btn.setObjectName("generate_solution_btn")
        self.generate_solution_btn.setVisible(True)

        # 添加货物操作按钮
        self.add_product_btn = QtWidgets.QPushButton(self.tool_widget)
        self.add_product_btn.setMinimumSize(QtCore.QSize(0, 30))
        self.add_product_btn.setObjectName("add_product_btn")
        self.add_product_btn.setVisible(True)

        self.remove_product_btn = QtWidgets.QPushButton(self.tool_widget)
        self.remove_product_btn.setMinimumSize(QtCore.QSize(0, 30))
        self.remove_product_btn.setObjectName("remove_product_btn")
        self.remove_product_btn.setVisible(True)

        self.import_product_btn = QtWidgets.QPushButton(self.tool_widget)
        self.import_product_btn.setMinimumSize(QtCore.QSize(0, 30))
        self.import_product_btn.setObjectName("import_product_btn")
        self.import_product_btn.setVisible(True)

        # 3D视图控制按钮
        self.rotate_view_btn = QtWidgets.QPushButton(self.tool_widget)
        self.rotate_view_btn.setMinimumSize(QtCore.QSize(0, 30))
        self.rotate_view_btn.setObjectName("rotate_view_btn")
        self.rotate_view_btn.setVisible(False)

        self.zoom_in_btn = QtWidgets.QPushButton(self.tool_widget)
        self.zoom_in_btn.setMinimumSize(QtCore.QSize(0, 30))
        self.zoom_in_btn.setObjectName("zoom_in_btn")
        self.zoom_in_btn.setVisible(False)

        self.zoom_out_btn = QtWidgets.QPushButton(self.tool_widget)
        self.zoom_out_btn.setMinimumSize(QtCore.QSize(0, 30))
        self.zoom_out_btn.setObjectName("zoom_out_btn")
        self.zoom_out_btn.setVisible(False)

        self.reset_view_btn = QtWidgets.QPushButton(self.tool_widget)
        self.reset_view_btn.setMinimumSize(QtCore.QSize(0, 30))
        self.reset_view_btn.setObjectName("reset_view_btn")
        self.reset_view_btn.setVisible(False)

        self.export_image_btn = QtWidgets.QPushButton(self.tool_widget)
        self.export_image_btn.setMinimumSize(QtCore.QSize(0, 30))
        self.export_image_btn.setObjectName("export_image_btn")
        self.export_image_btn.setVisible(False)

        # 仿真视图控制按钮
        self.play_btn = QtWidgets.QPushButton(self.tool_widget)
        self.play_btn.setObjectName("play_btn")
        self.play_btn.setText("播放")
        self.tool_layout.addWidget(self.play_btn)

        self.pause_btn = QtWidgets.QPushButton(self.tool_widget)
        self.pause_btn.setObjectName("pause_btn")
        self.pause_btn.setText("暂停")
        self.tool_layout.addWidget(self.pause_btn)

        self.stop_btn = QtWidgets.QPushButton(self.tool_widget)
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setText("停止")
        self.tool_layout.addWidget(self.stop_btn)

        self.export_video_btn = QtWidgets.QPushButton(self.tool_widget)
        self.export_video_btn.setObjectName("export_video_btn")
        self.export_video_btn.setText("导出视频")
        self.tool_layout.addWidget(self.export_video_btn)

        # 添加到布局
        self.tool_layout.addWidget(self.clear_condition_btn)
        self.tool_layout.addWidget(self.generate_solution_btn)
        # self.tool_layout.addWidget(QtWidgets.QLabel("货物操作:"))  # 添加分隔标签
        self.tool_layout.addWidget(self.add_product_btn)
        self.tool_layout.addWidget(self.remove_product_btn)
        self.tool_layout.addWidget(self.import_product_btn)
        # self.tool_layout.addWidget(QtWidgets.QLabel("3D控制:"))  # 添加分隔标签
        self.tool_layout.addWidget(self.rotate_view_btn)
        self.tool_layout.addWidget(self.zoom_in_btn)
        self.tool_layout.addWidget(self.zoom_out_btn)
        self.tool_layout.addWidget(self.reset_view_btn)
        self.tool_layout.addWidget(self.export_image_btn)

        # self.tool_layout.addItem(self.verticalSpacer)

        self.right_dock.setWidget(self.tool_widget)
        PackingSoftware.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.right_dock)

        # -------------------------------------------- 日志窗口 -------------------------------------------------------
        self.bottom_dock = QtWidgets.QDockWidget(PackingSoftware)
        self.bottom_dock.setObjectName("bottom_dock")
        self.bottom_dock.setAllowedAreas(
            QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.TopDockWidgetArea
        )

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setObjectName("log_text")
        self.log_text.setReadOnly(True)
        self.bottom_dock.setWidget(self.log_text)
        PackingSoftware.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.bottom_dock)

        self.retranslateUi(PackingSoftware)
        QtCore.QMetaObject.connectSlotsByName(PackingSoftware)

    def retranslateUi(self, PackingSoftware):
        _translate = QtCore.QCoreApplication.translate
        PackingSoftware.setWindowTitle(_translate("PackingSoftware", "智能装箱软件"))
        self.condition_form.setTitle(_translate("PackingSoftware", "装箱条件设置"))
        self.product_group.setTitle(_translate("PackingSoftware", "待装箱货物"))
        self.clear_condition_btn.setText(_translate("PackingSoftware", "清空条件"))
        self.generate_solution_btn.setText(_translate("PackingSoftware", "生成方案"))
        self.add_product_btn.setText(_translate("PackingSoftware", "添加货物"))
        self.remove_product_btn.setText(_translate("PackingSoftware", "移除货物"))
        self.import_product_btn.setText(_translate("PackingSoftware", "导入货物"))
        self.rotate_view_btn.setText(_translate("PackingSoftware", "旋转视图"))
        self.zoom_in_btn.setText(_translate("PackingSoftware", "放大"))
        self.zoom_out_btn.setText(_translate("PackingSoftware", "缩小"))
        self.reset_view_btn.setText(_translate("PackingSoftware", "重置视图"))
        self.export_image_btn.setText(_translate("PackingSoftware", "导出图片"))

        self.label_statement.setText(_translate("PackingSoftware", "报表界面"))
        self.btn_product.setText(_translate("PackingSoftware", "产品数据"))
        self.btn_pallet.setText(_translate("PackingSoftware", "托盘数据"))
        self.btn_container.setText(_translate("PackingSoftware", "集装箱数据"))
        self.left_dock.setWindowTitle(_translate("PackingSoftware", "功能导航"))
        self.right_dock.setWindowTitle(_translate("PackingSoftware", "工具栏"))
        self.bottom_dock.setWindowTitle(_translate("PackingSoftware", "日志"))
        self.clear_condition_btn.setText(_translate("PackingSoftware", "清空条件"))
        self.menu_file.setTitle(_translate("PackingSoftware", "文件"))
        self.menu_edit.setTitle(_translate("PackingSoftware", "编辑"))
        self.menu_view.setTitle(_translate("PackingSoftware", "视图"))
        self.menu_help.setTitle(_translate("PackingSoftware", "帮助"))
        self.menu_settings.setTitle(_translate("PackingSoftware", "设置"))
        self.action_new.setText(_translate("PackingSoftware", "新建"))
        self.action_open.setText(_translate("PackingSoftware", "打开"))
        self.action_save.setText(_translate("PackingSoftware", "保存"))
        self.action_exit.setText(_translate("PackingSoftware", "退出"))
        self.action_undo.setText(_translate("PackingSoftware", "撤销"))
        self.action_redo.setText(_translate("PackingSoftware", "重做"))
        self.action_toggle_left_dock.setText(
            _translate("PackingSoftware", "显示功能栏")
        )
        self.action_toggle_right_dock.setText(
            _translate("PackingSoftware", "显示工具栏")
        )
        self.action_toggle_bottom_dock.setText(
            _translate("PackingSoftware", "显示日志窗口")
        )
        self.action_about.setText(_translate("PackingSoftware", "关于"))
        self.action_settings.setText(_translate("PackingSoftware", "动画速度"))
        self.theme_settings.setText(_translate("PackingSoftware", "主题设置"))
        self.language_settings.setText(_translate("PackingSoftware", "语言设置"))
