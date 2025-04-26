"""
此文件用于设计用户图形交互界面
"""

import sys
import os
import tempfile

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)
from datetime import datetime
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QFile, QTextStream
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QAction,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QLineEdit,
    QMessageBox,
)
from core.algorithms import HybridOptimizer
from interface.qt_desgin.ui_main_window import Ui_PackingSoftware
from interface.gui.utils.resource_loader import get_icon, get_pixmap
from interface.gui.utils.style_manager import load_stylesheet
from interface.gui.widgets.enhanced_table import EnhancedTableWidget
from interface.gui.dialogs.about_dialog import AboutDialog
from interface.gui.dialogs.settings_dialog import SettingDialog
from interface.gui.widgets.result_3d_view import Result3DView
from interface.gui.widgets.report_view import ReportView
from interface.gui.widgets.simulation_view import SimulationView
from database.models import PackingReport, Product, Pallet, Container
from database.db_manager import DatabaseManager


class ProductSelectionDialog(QDialog):
    """产品选择对话框"""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("选择产品")
        self.setMinimumSize(600, 400)

        self.selected_products = {}  # {product_id: quantity}

        self.init_ui()
        self.load_products()

    def init_ui(self):
        self.layout = QtWidgets.QVBoxLayout(self)

        # 搜索框
        self.search_layout = QtWidgets.QHBoxLayout()
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("输入SKU或名称搜索...")
        self.search_btn = QtWidgets.QPushButton("搜索")
        self.search_btn.clicked.connect(self.search_products)
        self.search_layout.addWidget(self.search_input)
        self.search_layout.addWidget(self.search_btn)
        self.layout.addLayout(self.search_layout)

        # 产品表格
        self.product_table = QtWidgets.QTableWidget()
        self.product_table.setColumnCount(6)
        self.product_table.setHorizontalHeaderLabels(
            ["选择", "SKU", "名称", "尺寸(mm)", "重量(kg)", "数量"]
        )
        self.product_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.product_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.layout.addWidget(self.product_table)

        # 按钮
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def load_products(self, keyword=None):
        """加载产品数据"""
        self.product_table.setRowCount(0)

        try:
            query = self.session.query(Product)
            if keyword:
                query = query.filter(
                    (Product.sku.contains(keyword))
                    | (Product.item_name.contains(keyword))
                    | (Product.frgn_name.contains(keyword))
                )

            products = query.all()
            self.product_table.setRowCount(len(products))

            for row, product in enumerate(products):
                # 选择复选框
                check_box = QtWidgets.QCheckBox()
                check_box.setChecked(product.product_id in self.selected_products)
                check_widget = QtWidgets.QWidget()
                check_layout = QtWidgets.QHBoxLayout(check_widget)
                check_layout.addWidget(check_box)
                check_layout.setAlignment(QtCore.Qt.AlignCenter)
                check_layout.setContentsMargins(0, 0, 0, 0)
                self.product_table.setCellWidget(row, 0, check_widget)

                # SKU
                sku_item = QtWidgets.QTableWidgetItem(product.sku)
                sku_item.setData(QtCore.Qt.UserRole, product.product_id)
                self.product_table.setItem(row, 1, sku_item)

                # 名称
                name_item = QtWidgets.QTableWidgetItem(
                    product.item_name or product.frgn_name or ""
                )
                self.product_table.setItem(row, 2, name_item)

                # 尺寸
                size_item = QtWidgets.QTableWidgetItem(
                    f"{product.length}x{product.width}x{product.height}"
                )
                self.product_table.setItem(row, 3, size_item)

                # 重量
                weight_item = QtWidgets.QTableWidgetItem(str(product.weight))
                self.product_table.setItem(row, 4, weight_item)

                # 数量
                spin_box = QtWidgets.QSpinBox()
                spin_box.setMinimum(1)
                spin_box.setMaximum(1000)
                if product.product_id in self.selected_products:
                    spin_box.setValue(self.selected_products[product.product_id])
                self.product_table.setCellWidget(row, 5, spin_box)

                # 连接信号
                check_box.stateChanged.connect(
                    lambda state, r=row: self.toggle_product_selection(state, r)
                )

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载产品数据失败:\n{str(e)}")

    def search_products(self):
        """搜索产品"""
        keyword = self.search_input.text().strip()
        self.load_products(keyword)

    def toggle_product_selection(self, state, row):
        """切换产品选择状态"""
        product_id = self.product_table.item(row, 1).data(QtCore.Qt.UserRole)
        quantity = self.product_table.cellWidget(row, 5).value()

        if state == QtCore.Qt.Checked:
            self.selected_products[product_id] = quantity
        elif product_id in self.selected_products:
            del self.selected_products[product_id]

    def get_selected_products(self):
        """获取选择的产品及其数量"""
        result = {}
        try:
            for row in range(self.product_table.rowCount()):
                check_widget = self.product_table.cellWidget(row, 0)
                check_box = check_widget.findChild(QtWidgets.QCheckBox)

                if check_box and check_box.isChecked():
                    product_id = self.product_table.item(row, 1).data(
                        QtCore.Qt.UserRole
                    )
                    quantity = self.product_table.cellWidget(row, 5).value()
                    product = self.session.query(Product).get(product_id)
                    if product:
                        result[product] = quantity
        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取选择的产品失败:\n{str(e)}")

        return result


class DataEditDialog(QDialog):
    """通用数据编辑对话框"""

    def __init__(self, data_type, parent=None, record=None):
        super().__init__(parent)
        self.data_type = data_type
        self.record = record
        self.init_ui()
        self.setWindowTitle(f"编辑{data_type}数据")

    def init_ui(self):
        self.layout = QFormLayout()
        self.fields = {}

        if self.data_type == "product":
            self.init_product_fields()
        elif self.data_type == "pallet":
            self.init_pallet_fields()
        elif self.data_type == "container":
            self.init_container_fields()

        # 添加确定/取消按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addRow(buttons)
        self.setLayout(self.layout)

        # 如果是编辑模式，填充现有数据
        if self.record:
            self.populate_fields()

    def init_product_fields(self):
        """初始化产品字段"""
        self.fields["sku"] = QtWidgets.QLineEdit()
        self.fields["frgn_name"] = QtWidgets.QLineEdit()
        self.fields["item_name"] = QtWidgets.QLineEdit()
        self.fields["length"] = QtWidgets.QSpinBox()
        self.fields["width"] = QtWidgets.QSpinBox()
        self.fields["height"] = QtWidgets.QSpinBox()
        self.fields["weight"] = QtWidgets.QDoubleSpinBox()
        self.fields["direction"] = QtWidgets.QComboBox()
        self.fields["fragile"] = QtWidgets.QComboBox()

        # 设置范围和默认值
        for dim in ["length", "width", "height"]:
            self.fields[dim].setRange(1, 10000)
        self.fields["weight"].setRange(0.001, 10000)
        self.fields["direction"].addItems(["固定方向", "允许旋转"])
        self.fields["fragile"].addItems(["普通", "易碎", "非常易碎", "极度易碎"])

        # 添加到布局
        labels = [
            "SKU",
            "外文名称",
            "产品名称",
            "长度(mm)",
            "宽度(mm)",
            "高度(mm)",
            "重量(kg)",
            "放置方向",
            "易碎等级",
        ]
        for label, field in zip(labels, self.fields.values()):
            self.layout.addRow(label, field)

    def init_pallet_fields(self):
        """初始化托盘字段"""
        self.fields["length"] = QtWidgets.QSpinBox()
        self.fields["width"] = QtWidgets.QSpinBox()
        self.fields["height"] = QtWidgets.QSpinBox()
        self.fields["max_weight"] = QtWidgets.QDoubleSpinBox()

        for dim in ["length", "width", "height"]:
            self.fields[dim].setRange(1, 10000)
        self.fields["max_weight"].setRange(0.001, 10000)

        labels = ["长度(mm)", "宽度(mm)", "高度(mm)", "最大承重(kg)"]
        for label, field in zip(labels, self.fields.values()):
            self.layout.addRow(label, field)

    def init_container_fields(self):
        """初始化集装箱字段"""
        self.fields["name"] = QtWidgets.QLineEdit()
        self.fields["length"] = QtWidgets.QSpinBox()
        self.fields["width"] = QtWidgets.QSpinBox()
        self.fields["height"] = QtWidgets.QSpinBox()
        self.fields["max_weight"] = QtWidgets.QDoubleSpinBox()

        for dim in ["length", "width", "height"]:
            self.fields[dim].setRange(1, 10000)
        self.fields["max_weight"].setRange(0.001, 10000)

        labels = ["名称", "长度(mm)", "宽度(mm)", "高度(mm)", "最大承重(kg)"]
        for label, field in zip(labels, self.fields.values()):
            self.layout.addRow(label, field)

    def populate_fields(self):
        """填充现有数据到字段"""
        if self.data_type == "product":
            self.fields["sku"].setText(self.record.sku)
            self.fields["frgn_name"].setText(self.record.frgn_name or "")
            self.fields["item_name"].setText(self.record.item_name or "")
            self.fields["length"].setValue(self.record.length)
            self.fields["width"].setValue(self.record.width)
            self.fields["height"].setValue(self.record.height)
            self.fields["weight"].setValue(float(self.record.weight))
            self.fields["direction"].setCurrentIndex(self.record.direction)
            self.fields["fragile"].setCurrentIndex(self.record.fragile)
        elif self.data_type == "pallet":
            self.fields["length"].setValue(self.record.length)
            self.fields["width"].setValue(self.record.width)
            self.fields["height"].setValue(self.record.height)
            self.fields["max_weight"].setValue(float(self.record.max_weight))
        elif self.data_type == "container":
            self.fields["name"].setText(self.record.name or "")
            self.fields["length"].setValue(self.record.length)
            self.fields["width"].setValue(self.record.width)
            self.fields["height"].setValue(self.record.height)
            self.fields["max_weight"].setValue(float(self.record.max_weight))

    def get_data(self):
        """获取输入数据为字典"""
        data = {}
        if self.data_type == "product":
            data = {
                "sku": self.fields["sku"].text().strip(),
                "frgn_name": self.fields["frgn_name"].text().strip(),
                "item_name": self.fields["item_name"].text().strip(),
                "length": self.fields["length"].value(),
                "width": self.fields["width"].value(),
                "height": self.fields["height"].value(),
                "weight": str(self.fields["weight"].value()),
                "direction": self.fields["direction"].currentIndex(),
                "fragile": self.fields["fragile"].currentIndex(),
            }
        elif self.data_type == "pallet":
            data = {
                "length": self.fields["length"].value(),
                "width": self.fields["width"].value(),
                "height": self.fields["height"].value(),
                "max_weight": str(self.fields["max_weight"].value()),
            }
        elif self.data_type == "container":
            data = {
                "name": self.fields["name"].text().strip(),
                "length": self.fields["length"].value(),
                "width": self.fields["width"].value(),
                "height": self.fields["height"].value(),
                "max_weight": str(self.fields["max_weight"].value()),
            }
        return data

    def accept(self):
        """重写accept方法，添加数据验证"""
        if self.data_type == "product":
            sku = self.fields["sku"].text().strip()
            if not sku:
                QMessageBox.warning(self, "警告", "SKU不能为空")
                return

            length = self.fields["length"].value()
            width = self.fields["width"].value()
            height = self.fields["height"].value()
            if length <= 0 or width <= 0 or height <= 0:
                QMessageBox.warning(self, "警告", "尺寸必须大于0")
                return

        super().accept()


class PackingSoftware(QMainWindow, Ui_PackingSoftware):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 工具按钮属性声明
        self.add_btn = None
        self.delete_btn = None
        self.edit_btn = None
        self.search_btn = None
        self.import_btn = None
        self.export_btn = None
        self.search_input = None
        self.current_tools = []

        # 初始化数据库
        self.db_manager = DatabaseManager()
        self.session = self.db_manager.get_session()

        # 主窗口尺寸控制
        self.resize(1000, 600)  # 默认大小
        self.setMinimumSize(800, 500)  # 最小大小

        # 设置主窗口中央部件的边距
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 数据管理相关属性
        self.current_data_type = "product"
        self.data_models = {
            "product": (
                Product,
                [
                    "ID",
                    "SKU",
                    "外文名称",
                    "产品名称",
                    "长(mm)",
                    "宽(mm)",
                    "高(mm)",
                    "重量(kg)",
                    "方向",
                    "易碎度",
                ],
            ),
            "pallet": (Pallet, ["ID", "长(mm)", "宽(mm)", "高(mm)", "最大承重(kg)"]),
            "container": (
                Container,
                ["ID", "名称", "长(mm)", "宽(mm)", "高(mm)", "最大承重(kg)"],
            ),
        }

        # 初始化3D视图
        self.result_3d_view = Result3DView(self.page_3dview)
        old_container = self.page_3dview.findChild(
            QtWidgets.QWidget, "view3d_container"
        )
        if old_container:
            self.view3d_layout.replaceWidget(old_container, self.result_3d_view)
            old_container.deleteLater()
        else:
            self.view3d_layout.insertWidget(0, self.result_3d_view, 1)

        # 初始化仿真视图
        self.page_simulation = SimulationView()
        self.page_simulation.setObjectName("page_simulation")
        self.stacked_widget.addWidget(self.page_simulation)

        # 初始化报表视图
        self.report_view = ReportView(self.page_statement)
        old_statement = self.page_statement.findChild(
            QtWidgets.QLabel, "label_statement"
        )
        if old_statement:
            self.statement_layout.replaceWidget(old_statement, self.report_view)
            old_statement.deleteLater()
        else:
            self.statement_layout.addWidget(self.report_view, 1)

        # 连接控制按钮信号
        self.rotate_view_btn.clicked.connect(self.result_3d_view.toggle_rotation)
        self.zoom_in_btn.clicked.connect(self.result_3d_view.zoom_in)
        self.zoom_out_btn.clicked.connect(self.result_3d_view.zoom_out)
        self.reset_view_btn.clicked.connect(self.result_3d_view.reset_view)
        self.export_image_btn.clicked.connect(self.result_3d_view.export_image)
        self.play_btn.clicked.connect(self.page_simulation.toggle_playback)
        self.pause_btn.clicked.connect(self.page_simulation.pause_playback)
        self.stop_btn.clicked.connect(self.page_simulation.stop_playback)
        self.export_video_btn.clicked.connect(self.page_simulation.export_video)
        self.function_list.currentRowChanged.connect(self.on_page_changed)
        self.generate_solution_btn.clicked.connect(self.pass_solution_to_simulation)

        self.load_styles()
        self.init_components()
        load_stylesheet(self)
        self.init_ui()
        self.connect_signals()

        # 初始状态
        self.switch_page(0)  # 默认显示条件设置界面
        self.init_condition_page()
        self.switch_data_type("product")
        self.log_message("应用程序启动完成")

    def load_styles(self):
        """加载样式表"""
        qss_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "styles",
            "main_style.qss",
        )
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())
        else:
            print(f"警告：未找到样式文件 {qss_path}")

    def init_components(self):
        """初始化UI组件"""
        # 数据表格
        self.data_table = EnhancedTableWidget(self.page_data)
        old_table = self.page_data.findChild(QtWidgets.QTableWidget)
        if old_table:
            self.data_layout.replaceWidget(old_table, self.data_table)
            old_table.deleteLater()
        else:
            self.data_layout.addWidget(self.data_table, 1)

        # 设置最小尺寸限制
        self.left_dock.setMinimumWidth(100)
        self.right_dock.setMinimumWidth(150)  # 加宽工具栏
        self.right_dock.setMaximumWidth(180)
        self.bottom_dock.setMinimumHeight(80)

        # 工具栏样式
        self.tool_widget.setStyleSheet(
            """
            QWidget#tool_widget {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton {
                min-height: 30px;
                max-height: 30px;
                margin: 2px;
            }
            QLineEdit {
            min-height: 25px;
            max-height: 25px;
            }
        """
        )

        # 窗口属性
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMaximizeButtonHint)

    def is_button_valid(self, btn):
        """安全检查按钮是否有效"""
        try:
            return btn is not None and btn.isWidgetType() and btn.parent() is not None
        except RuntimeError:
            return False

    def add_tool_components(self):
        """添加数据管理工具栏组件"""
        # 只在数据管理页面添加工具
        if self.stacked_widget.currentIndex() != 4:
            return

        # 清除现有工具
        self.clear_tools()

        # 确保 tool_layout 存在
        if not hasattr(self, "tool_layout"):
            self.tool_layout = QVBoxLayout()
            self.tool_widget.setLayout(self.tool_layout)

        # 创建工具按钮
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("输入查询条件...")
        self.tool_layout.addWidget(self.search_input)

        self.search_btn = QtWidgets.QPushButton("查询")
        self.tool_layout.addWidget(self.search_btn)
        self.search_btn.clicked.connect(self.search_records)

        self.add_btn = QtWidgets.QPushButton("添加")
        self.tool_layout.addWidget(self.add_btn)
        self.add_btn.clicked.connect(self.add_record)

        self.delete_btn = QtWidgets.QPushButton("删除")
        self.tool_layout.addWidget(self.delete_btn)
        self.delete_btn.clicked.connect(self.delete_record)

        self.edit_btn = QtWidgets.QPushButton("修改")
        self.tool_layout.addWidget(self.edit_btn)
        self.edit_btn.clicked.connect(self.edit_record)

        self.import_btn = QtWidgets.QPushButton("从表格导入")
        self.tool_layout.addWidget(self.import_btn)
        self.import_btn.clicked.connect(self.import_from_excel)

        self.export_btn = QtWidgets.QPushButton("从表格导出")
        self.tool_layout.addWidget(self.export_btn)
        self.export_btn.clicked.connect(self.export_to_excel)

        # 记录当前工具
        self.current_tools = [
            self.search_input,
            self.search_btn,
            self.add_btn,
            self.delete_btn,
            self.edit_btn,
            self.import_btn,
            self.export_btn,
        ]

    def clear_tools(self):
        """清除当前工具按钮"""
        if hasattr(self, "tool_layout"):
            for tool in self.current_tools:
                if tool is not None and tool.parent() is not None:
                    self.tool_layout.removeWidget(tool)
                    tool.setParent(None)
                    tool.deleteLater()
            self.current_tools = []

    def log_message(self, message):
        """向日志窗口添加带时间戳的消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def init_ui(self):
        # 组织工具按钮组
        self.tool_groups = {
            0: [  # 条件设置界面
                self.clear_condition_btn,
                self.generate_solution_btn,
                self.add_product_btn,
                self.remove_product_btn,
                self.import_product_btn,
            ],
            1: [],  # 3D展示界面
            2: [],  # 动态仿真界面
            3: [],  # 报表界面
            4: [],  # 数据管理界面(工具由add_tool_components动态添加)
        }

        # 设置初始可见性
        self.update_tools(0)  # 默认显示第一个页面的工具

        # 设置停靠窗口初始大小
        self.left_dock.resize(150, self.height())
        self.right_dock.resize(200, self.height())
        self.bottom_dock.resize(self.width(), 100)

        # 设置最小尺寸限制
        self.left_dock.setMinimumWidth(100)
        self.right_dock.setMinimumWidth(150)
        self.bottom_dock.setMinimumHeight(80)

    def connect_signals(self):
        # 连接功能列表切换信号
        self.function_list.currentRowChanged.connect(self.switch_page)

        # 连接视图菜单动作
        self.action_toggle_left_dock.triggered.connect(self.toggle_left_dock)
        self.action_toggle_right_dock.triggered.connect(self.toggle_right_dock)
        self.action_toggle_bottom_dock.triggered.connect(self.toggle_bottom_dock)

        # 连接文件菜单动作
        self.action_exit.triggered.connect(self.close)

        # 连接其他菜单动作
        self.action_about.triggered.connect(self.show_about_dialog)
        self.action_settings.triggered.connect(self.show_settings_dialog)

        # 连接生成方案按钮
        self.generate_solution_btn.clicked.connect(self.calculate_packing)

        # 数据切换按钮
        self.btn_product.clicked.connect(lambda: self.switch_data_type("product"))
        self.btn_pallet.clicked.connect(lambda: self.switch_data_type("pallet"))
        self.btn_container.clicked.connect(lambda: self.switch_data_type("container"))

    def switch_page(self, index):
        """切换功能页面"""
        self.stacked_widget.setCurrentIndex(index)
        self.update_tools(index)

        # 如果是数据管理页面，刷新数据并添加工具
        if index == 4:
            self.load_data()
            self.add_tool_components()

    def switch_data_type(self, data_type):
        """切换数据类型"""
        self.current_data_type = data_type
        self.update_table_headers()
        self.load_data()

    def update_table_headers(self):
        """更新表格列头"""
        _, headers = self.data_models[self.current_data_type]
        self.data_table.setColumnCount(len(headers))
        self.data_table.setHorizontalHeaderLabels(headers)
        self.data_table.resizeColumnsToContents()

    def load_data(self):
        """从数据库加载数据"""
        model, headers = self.data_models[self.current_data_type]

        try:
            # 设置表头
            self.data_table.setColumnCount(len(headers))
            self.data_table.setHorizontalHeaderLabels(headers)

            # 清空表格
            self.data_table.setRowCount(0)

            records = self.session.query(model).all()
            self.data_table.setRowCount(len(records))

            for row_idx, record in enumerate(records):
                if isinstance(record, Product):
                    self.populate_product_row(row_idx, record)
                elif isinstance(record, Pallet):
                    self.populate_pallet_row(row_idx, record)
                elif isinstance(record, Container):
                    self.populate_container_row(row_idx, record)

            # 设置单元格居中显示
            for row in range(self.data_table.rowCount()):
                for col in range(self.data_table.columnCount()):
                    item = self.data_table.item(row, col)
                    if item:  # 确保item存在
                        item.setTextAlignment(Qt.AlignCenter)

            # 设置表头自适应策略
            header = self.data_table.horizontalHeader()
            header.setDefaultSectionSize(100)  # 默认列宽
            header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)  # 所有列平均拉伸

            # 允许用户手动调整列宽
            header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
            header.setCascadingSectionResizes(True)

            # 首次加载时自动调整列宽
            self.data_table.resizeColumnsToContents()

            # 设置最小列宽，防止内容被压缩
            for col in range(self.data_table.columnCount()):
                header.setMinimumSectionSize(60)

            self.log_message(f"已加载 {len(records)} 条{self.current_data_type}记录")
        except Exception as e:
            self.log_message(f"加载数据错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载数据时发生错误:\n{str(e)}")

    def populate_product_row(self, row_idx, product):
        """填充产品数据行"""

        def create_centered_item(text):
            item = QtWidgets.QTableWidgetItem(str(text))
            item.setTextAlignment(Qt.AlignCenter)
            return item

        self.data_table.setItem(
            row_idx, 0, create_centered_item(str(product.product_id))
        )
        self.data_table.setItem(row_idx, 1, create_centered_item(product.sku))
        self.data_table.setItem(
            row_idx, 2, create_centered_item(product.frgn_name or "")
        )
        self.data_table.setItem(
            row_idx, 3, create_centered_item(product.item_name or "")
        )
        self.data_table.setItem(row_idx, 4, create_centered_item(str(product.length)))
        self.data_table.setItem(row_idx, 5, create_centered_item(str(product.width)))
        self.data_table.setItem(row_idx, 6, create_centered_item(str(product.height)))
        self.data_table.setItem(row_idx, 7, create_centered_item(str(product.weight)))
        self.data_table.setItem(
            row_idx,
            8,
            create_centered_item("允许旋转" if product.direction else "固定方向"),
        )
        fragile_levels = ["普通", "易碎", "非常易碎", "极度易碎"]
        self.data_table.setItem(
            row_idx, 9, create_centered_item(fragile_levels[product.fragile])
        )

    def populate_pallet_row(self, row_idx, pallet):
        """填充托盘数据行"""

        def create_centered_item(text):
            item = QtWidgets.QTableWidgetItem(str(text))
            item.setTextAlignment(Qt.AlignCenter)
            return item

        self.data_table.setItem(row_idx, 0, create_centered_item(str(pallet.pallet_id)))
        self.data_table.setItem(row_idx, 1, create_centered_item(str(pallet.length)))
        self.data_table.setItem(row_idx, 2, create_centered_item(str(pallet.width)))
        self.data_table.setItem(row_idx, 3, create_centered_item(str(pallet.height)))
        self.data_table.setItem(
            row_idx, 4, create_centered_item(str(pallet.max_weight))
        )

    def populate_container_row(self, row_idx, container):
        """填充集装箱数据行"""

        def create_centered_item(text):
            item = QtWidgets.QTableWidgetItem(str(text))
            item.setTextAlignment(Qt.AlignCenter)
            return item

        self.data_table.setItem(
            row_idx, 0, create_centered_item(str(container.container_id))
        )
        self.data_table.setItem(row_idx, 1, create_centered_item(container.name or ""))
        self.data_table.setItem(row_idx, 2, create_centered_item(str(container.length)))
        self.data_table.setItem(row_idx, 3, create_centered_item(str(container.width)))
        self.data_table.setItem(row_idx, 4, create_centered_item(str(container.height)))
        self.data_table.setItem(
            row_idx, 5, create_centered_item(str(container.max_weight))
        )

    def add_record(self):
        """添加记录"""
        dialog = DataEditDialog(self.current_data_type, self)
        if dialog.exec_() == QDialog.Accepted:
            try:
                data = dialog.get_data()
                model, _ = self.data_models[self.current_data_type]

                if self.current_data_type == "product":
                    new_record = Product(
                        sku=data["sku"],
                        frgn_name=data["frgn_name"],
                        item_name=data["item_name"],
                        length=data["length"],
                        width=data["width"],
                        height=data["height"],
                        weight=data["weight"],
                        direction=data["direction"],
                        fragile=data["fragile"],
                    )
                elif self.current_data_type == "pallet":
                    new_record = Pallet(
                        length=data["length"],
                        width=data["width"],
                        height=data["height"],
                        max_weight=data["max_weight"],
                    )
                elif self.current_data_type == "container":
                    new_record = Container(
                        name=data["name"],
                        length=data["length"],
                        width=data["width"],
                        height=data["height"],
                        max_weight=data["max_weight"],
                    )

                self.session.add(new_record)
                self.session.commit()
                self.load_data()
                self.log_message(f"成功添加{self.current_data_type}记录")
            except Exception as e:
                self.session.rollback()
                self.log_message(f"添加记录失败: {str(e)}")
                QMessageBox.critical(self, "错误", f"添加记录时发生错误:\n{str(e)}")

    def edit_record(self):
        """编辑记录"""
        selected = self.data_table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.warning(self, "警告", "请先选择要编辑的记录")
            return

        if len(selected) > 1:
            QMessageBox.warning(self, "警告", "每次只能编辑一条记录")
            return

        row = selected[0].row()
        record_id = int(self.data_table.item(row, 0).text())

        try:
            model, _ = self.data_models[self.current_data_type]
            record = self.session.query(model).get(record_id)

            if not record:
                QMessageBox.warning(self, "警告", "找不到选中的记录")
                return

            dialog = DataEditDialog(self.current_data_type, self, record)
            if dialog.exec_() == QDialog.Accepted:
                data = dialog.get_data()

                if self.current_data_type == "product":
                    record.sku = data["sku"]
                    record.frgn_name = data["frgn_name"]
                    record.item_name = data["item_name"]
                    record.length = data["length"]
                    record.width = data["width"]
                    record.height = data["height"]
                    record.weight = data["weight"]
                    record.direction = data["direction"]
                    record.fragile = data["fragile"]
                elif self.current_data_type == "pallet":
                    record.length = data["length"]
                    record.width = data["width"]
                    record.height = data["height"]
                    record.max_weight = data["max_weight"]
                elif self.current_data_type == "container":
                    record.name = data["name"]
                    record.length = data["length"]
                    record.width = data["width"]
                    record.height = data["height"]
                    record.max_weight = data["max_weight"]

                self.session.commit()
                self.load_data()
                self.log_message(
                    f"成功更新{self.current_data_type}记录(ID: {record_id})"
                )
        except Exception as e:
            self.session.rollback()
            self.log_message(f"编辑记录失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"编辑记录时发生错误:\n{str(e)}")

    def delete_record(self):
        """删除记录"""
        selected = self.data_table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.warning(self, "警告", "请先选择要删除的记录")
            return

        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除选中的 {len(selected)} 条记录吗?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.No:
            return

        try:
            model, _ = self.data_models[self.current_data_type]
            deleted_ids = []

            for index in reversed(selected):
                row = index.row()
                record_id = int(self.data_table.item(row, 0).text())
                record = self.session.query(model).get(record_id)

                if record:
                    self.session.delete(record)
                    deleted_ids.append(str(record_id))

            self.session.commit()
            self.load_data()
            self.log_message(f"成功删除记录(ID: {', '.join(deleted_ids)})")
        except Exception as e:
            self.session.rollback()
            self.log_message(f"删除记录失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"删除记录时发生错误:\n{str(e)}")

    def search_records(self):
        """查询记录"""
        keyword = self.search_input.text().strip()
        if not keyword:
            self.load_data()
            return

        model, headers = self.data_models[self.current_data_type]

        try:
            self.data_table.setRowCount(0)

            if self.current_data_type == "product":
                query = self.session.query(model).filter(
                    (model.sku.contains(keyword))
                    | (model.frgn_name.contains(keyword))
                    | (model.item_name.contains(keyword))
                )
            elif self.current_data_type == "container":
                query = self.session.query(model).filter(model.name.contains(keyword))
            else:
                query = self.session.query(model)

            records = query.all()
            self.data_table.setRowCount(len(records))

            for row_idx, record in enumerate(records):
                if isinstance(record, Product):
                    self.populate_product_row(row_idx, record)
                elif isinstance(record, Pallet):
                    self.populate_pallet_row(row_idx, record)
                elif isinstance(record, Container):
                    self.populate_container_row(row_idx, record)

            self.log_message(f"找到 {len(records)} 条匹配'{keyword}'的记录")
        except Exception as e:
            self.log_message(f"查询错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"查询时发生错误:\n{str(e)}")

    def import_from_excel(self):
        """从Excel导入数据"""
        QMessageBox.information(self, "提示", "Excel导入功能将在后续版本实现")
        self.log_message("Excel导入功能将在后续版本实现")

    def export_to_excel(self):
        """导出数据到Excel"""
        QMessageBox.information(self, "提示", "Excel导出功能将在后续版本实现")
        self.log_message("Excel导出功能将在后续版本实现")

    # ------------------------------------------- 条件设置相关函数 -------------------------------------------
    def init_condition_page(self):
        """初始化条件设置页面"""
        # 连接信号
        self.pallet_check.stateChanged.connect(self.toggle_pallet_selection)
        self.add_product_btn.clicked.connect(self.add_product_to_list)
        self.remove_product_btn.clicked.connect(self.remove_product_from_list)
        self.import_product_btn.clicked.connect(self.import_products_from_excel)

        # 加载集装箱和托盘数据
        self.load_containers()
        self.load_pallets()

        # 初始化货物表格
        self.product_table.setColumnCount(5)
        self.product_table.setHorizontalHeaderLabels(
            ["SKU", "名称", "数量", "尺寸(mm)", "重量(kg)"]
        )
        self.product_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )

        # 设置表格样式
        self.product_table.setStyleSheet(
            """
            QTableWidget {
                border: 1px solid #ddd;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #ddd;
            }
        """
        )

    def calculate_packing(self):
        """执行装箱计算"""
        try:
            # 获取装箱条件
            conditions = self.get_packing_conditions()

            # 调用算法计算
            algorithm = HybridOptimizer(
                container=self.get_selected_container(),
                products=self.get_selected_products(),
                candidate_pallets=(
                    self.get_selected_pallets() if self.pallet_check.isChecked() else []
                ),
            )

            solution = algorithm.optimize()

            # 显示结果
            self.show_packing_result(solution)

        except Exception as e:
            self.log_message(f"计算失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"计算过程中发生错误:\n{str(e)}")

    def load_containers(self):
        """加载集装箱数据到下拉框"""
        self.container_combo.clear()
        try:
            containers = self.session.query(Container).all()
            for container in containers:
                self.container_combo.addItem(
                    f"{container.name} ({container.length}x{container.width}x{container.height}mm, {container.max_weight}kg)",
                    container.container_id,
                )
        except Exception as e:
            self.log_message(f"加载集装箱数据失败: {str(e)}")

    def load_pallets(self):
        """加载托盘数据到下拉框"""
        self.pallet_combo.clear()
        try:
            pallets = self.session.query(Pallet).all()
            for pallet in pallets:
                self.pallet_combo.addItem(
                    f"{pallet.length}x{pallet.width}x{pallet.height}mm, {pallet.max_weight}kg",
                    pallet.pallet_id,
                )
        except Exception as e:
            self.log_message(f"加载托盘数据失败: {str(e)}")

    def toggle_pallet_selection(self, state):
        """切换托盘选择可用状态"""
        self.pallet_combo.setEnabled(state == QtCore.Qt.Checked)

    def add_product_to_list(self):
        """从数据库添加产品到货物列表"""
        dialog = ProductSelectionDialog(self.session, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_products = dialog.get_selected_products()
            for product, quantity in selected_products.items():
                self.add_product_row(product, quantity)
            self.log_message(f"添加了 {len(selected_products)} 种产品到装箱列表")

    def add_product_row(self, product, quantity):
        """添加产品行到表格"""
        row = self.product_table.rowCount()
        self.product_table.insertRow(row)

        # SKU
        sku_item = QtWidgets.QTableWidgetItem(product.sku)
        sku_item.setData(QtCore.Qt.UserRole, product.product_id)  # 保存产品ID
        self.product_table.setItem(row, 0, sku_item)

        # 名称
        name_item = QtWidgets.QTableWidgetItem(
            product.item_name or product.frgn_name or ""
        )
        self.product_table.setItem(row, 1, name_item)

        # 数量
        spin_box = QtWidgets.QSpinBox()
        spin_box.setMinimum(1)
        spin_box.setMaximum(10000)
        spin_box.setValue(quantity)
        self.product_table.setCellWidget(row, 2, spin_box)

        # 尺寸
        size_item = QtWidgets.QTableWidgetItem(
            f"{product.length}x{product.width}x{product.height}"
        )
        self.product_table.setItem(row, 3, size_item)

        # 重量
        weight_item = QtWidgets.QTableWidgetItem(str(product.weight))
        self.product_table.setItem(row, 4, weight_item)

    def remove_product_from_list(self):
        """从货物列表中移除选中产品"""
        selected = self.product_table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.warning(self, "警告", "请先选择要移除的产品")
            return

        # 从后往前删除避免索引问题
        for index in reversed(selected):
            self.product_table.removeRow(index.row())

        self.log_message(f"移除了 {len(selected)} 个产品")

    def import_products_from_excel(self):
        """从Excel导入产品"""
        QMessageBox.information(self, "提示", "Excel导入功能将在后续版本实现")
        self.log_message("Excel导入功能将在后续版本实现")

    def get_packing_conditions(self):
        """获取当前设置的条件"""
        conditions = {
            "container_id": self.container_combo.currentData(),
            "use_pallet": self.pallet_check.isChecked(),
            "pallet_id": (
                self.pallet_combo.currentData()
                if self.pallet_check.isChecked()
                else None
            ),
            "transport_type": self.transport_combo.currentText(),
            "products": [],
        }

        # 收集产品信息
        for row in range(self.product_table.rowCount()):
            product_id = self.product_table.item(row, 0).data(QtCore.Qt.UserRole)
            quantity = self.product_table.cellWidget(row, 2).value()
            conditions["products"].append(
                {"product_id": product_id, "quantity": quantity}
            )

        return conditions

    def update_tools(self, index):
        """更新工具栏按钮显示"""
        # 隐藏所有工具按钮
        all_buttons = [
            self.clear_condition_btn,
            self.generate_solution_btn,
            self.add_product_btn,
            self.remove_product_btn,
            self.import_product_btn,
            self.rotate_view_btn,
            self.zoom_in_btn,
            self.zoom_out_btn,
            self.reset_view_btn,
            self.export_image_btn,
            self.play_btn,
            self.pause_btn,
            self.stop_btn,
            self.export_video_btn,
        ]

        for btn in all_buttons:
            if btn is not None:
                btn.setVisible(False)

        # 清除数据管理工具
        self.clear_tools()

        # 显示当前页面对应的工具按钮
        if index == 0:  # 条件设置页面
            self.clear_condition_btn.setVisible(True)
            self.generate_solution_btn.setVisible(True)
            self.add_product_btn.setVisible(True)
            self.remove_product_btn.setVisible(True)
            self.import_product_btn.setVisible(True)
        elif index == 1:  # 3D展示页面
            self.rotate_view_btn.setVisible(True)
            self.zoom_in_btn.setVisible(True)
            self.zoom_out_btn.setVisible(True)
            self.reset_view_btn.setVisible(True)
            self.export_image_btn.setVisible(True)
        elif index == 2:  # 仿真演示页面
            self.play_btn.setVisible(True)
            self.pause_btn.setVisible(True)
            self.stop_btn.setVisible(True)
            self.export_video_btn.setVisible(True)
        elif index == 4:  # 数据管理页面
            self.add_tool_components()

    def import_packing_conditions(self):
        """导入装箱条件"""
        QMessageBox.information(self, "提示", "条件导入功能将在后续版本实现")
        self.log_message("条件导入功能将在后续版本实现")

    def export_packing_conditions(self):
        """导出装箱条件"""
        conditions = self.get_packing_conditions()
        if not conditions["container_id"]:
            QMessageBox.warning(self, "警告", "请先选择集装箱")
            return

        if not conditions["products"]:
            QMessageBox.warning(self, "警告", "请添加至少一个产品")
            return

        # 这里添加实际导出逻辑
        QMessageBox.information(self, "提示", "条件导出功能将在后续版本实现")
        self.log_message("条件导出功能将在后续版本实现")

    def clear_packing_conditions(self):
        """清空装箱条件"""
        reply = QMessageBox.question(
            self,
            "确认清空",
            "确定要清空所有装箱条件吗?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.container_combo.setCurrentIndex(0)
            self.pallet_check.setChecked(False)
            self.transport_combo.setCurrentIndex(0)
            self.product_table.setRowCount(0)
            self.log_message("已清空装箱条件")

    # ------------------------------------------- 条件设置相关终止 -------------------------------------------

    # ------------------------------------------- 3D设置相关函数 -------------------------------------------
    def show_packing_result(self, solution):
        """显示装箱结果"""
        try:
            # 切换到3D视图
            self.stacked_widget.setCurrentIndex(1)

            # 确保result_3d_view正确初始化
            if (
                not hasattr(self.result_3d_view, "main_splitter")
                or not self.result_3d_view.main_splitter
            ):
                raise Exception("3D视图的分割器未正确初始化")

            # 确保solution包含必要的数据
            enhanced_solution = {
                **solution,
                "solution_id": solution.get(
                    "solution_id", "B" + datetime.now().strftime("%y%m%d%H%M")
                ),
                "products": [
                    {
                        **p,
                        "name": p.get("name", p.get("sku", f"产品{idx}")),
                        "rotatable": p.get("direction", 1) == 1,  # 1表示允许旋转
                        "load_bearing": p.get("fragile", 0) == 0,  # 非易碎品可以承重
                    }
                    for idx, p in enumerate(solution.get("products", []))
                ],
            }

            # 动态调整分割器比例
            self.adjust_splitter_proportion()

            # 调整分割器比例（确保3D视图占2/3宽度）
            splitter = self.result_3d_view.main_splitter
            if splitter:
                total_width = splitter.width()
                splitter.setSizes([total_width // 3, total_width * 2 // 3])

            # 更新状态栏
            total_items = sum(
                p.get("quantity", 1) for p in enhanced_solution.get("products", [])
            )
            total_volume = sum(
                (
                    p["dimensions"][0]
                    * p["dimensions"][1]
                    * p["dimensions"][2]
                    * p.get("quantity", 1)
                )
                / 1e9
                for p in enhanced_solution.get("products", [])
            )
            total_weight = sum(
                p.get("weight", 0) * p.get("quantity", 1)
                for p in enhanced_solution.get("products", [])
            )
            utilization = enhanced_solution.get("utilization", 0)
            stability = enhanced_solution.get("stability", 0)

            self.statusbar.showMessage(
                f"装箱完成 - 方案: {enhanced_solution['solution_id']} | "
                f"总件数: {total_items} | 利用率: {utilization:.1%} | "
                f"稳定性: {stability:.1%}"
            )

            # 记录到日志
            self.log_message(
                f"显示装箱结果 - 方案: {enhanced_solution['solution_id']}, "
                f"{total_items}件, {total_volume:.2f}m³, {total_weight:.2f}kg"
            )

        except Exception as e:
            self.log_message(f"显示装箱结果时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"显示装箱结果时发生错误:\n{str(e)}")

        # 保存到报表系统
        self.save_packing_result(enhanced_solution)

    def adjust_splitter_proportion(self):
        """动态调整分割器比例，确保3D视图占2/3宽度"""
        splitter = self.result_3d_view.main_splitter
        if splitter:
            # 更新分割器大小策略，确保左右面板按照2/3比例分配
            total_width = splitter.width()
            left_width = total_width // 3
            right_width = total_width * 2 // 3
            splitter.setSizes([left_width, right_width])
        else:
            self.log_message("警告：未找到3D视图的分割器")

    def resizeEvent(self, event):
        """重写窗口调整大小事件，动态更新分割器比例"""
        super(PackingSoftware, self).resizeEvent(event)
        # 窗口大小改变时，重新调整3D视图比例
        if hasattr(self, "result_3d_view"):
            self.adjust_splitter_proportion()

    # ------------------------------------------- 3D设置相关终止 -------------------------------------------

    # ------------------------------------------- 仿真界面相关函数 -------------------------------------------
    def on_page_changed(self, index):
        """切换页面时更新仿真视图的解决方案"""
        if index == 2:  # 动态仿真界面
            if hasattr(self, "current_solution"):
                self.simulation_view.set_solution(self.current_solution)

    def pass_solution_to_simulation(self):
        """将计算结果传递给仿真视图"""
        solution = self.calculate_packing()
        self.current_solution = solution
        self.simulation_view.set_solution(solution)

    # ------------------------------------------- 仿真界面相关终止 -------------------------------------------
    # ------------------------------------------- 报表界面相关设置 -------------------------------------------
    def save_packing_report(self, solution):
        """保存装箱方案到报表系统"""
        if not self.session:
            return

        try:
            # 创建新报表
            report = PackingReport(
                solution_id=solution.get(
                    "solution_id", f"B{datetime.now().strftime('%y%m%d%H%M')}"
                ),
                container_id=solution["container"].container_id,
                total_items=sum(
                    p.get("quantity", 1) for p in solution.get("products", [])
                ),
                total_volume=sum(
                    (
                        p["dimensions"][0]
                        * p["dimensions"][1]
                        * p["dimensions"][2]
                        * p.get("quantity", 1)
                    )
                    / 1e9
                    for p in solution.get("products", [])
                ),
                total_weight=sum(
                    p.get("weight", 0) * p.get("quantity", 1)
                    for p in solution.get("products", [])
                ),
                utilization=solution.get("utilization", 0),
                stability=solution.get("stability", 0),
                notes="自动生成的装箱方案",
            )

            # 保存3D视图截图
            img_path = os.path.join(tempfile.gettempdir(), f"{report.solution_id}.png")
            img = self.result_3d_view.view3d.grabFramebuffer()
            img.save(img_path)
            report.image_path = img_path

            self.session.add(report)
            self.session.commit()

            self.log_message(f"已保存方案 {report.solution_id} 到报表系统")
            return True

        except Exception as e:
            self.session.rollback()
            self.log_message(f"保存报表失败: {str(e)}")
            return False

    # ------------------------------------------- 报表界面相关终止 -------------------------------------------

    def closeEvent(self, event):
        """关闭窗口事件处理"""
        self.session.close()
        super().closeEvent(event)

    def show_about_dialog(self):
        dialog = AboutDialog(self)
        dialog.exec_()

    def show_settings_dialog(self):
        dialog = SettingDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.update_animation_speed(dialog.animation_speed.value())

    def update_animation_speed(self, speed):
        """更新动画速度"""
        self.log_message(f"动画速度设置为: {speed}")

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

    def toggle_bottom_dock(self):
        """切换底部日志窗口显示"""
        visible = not self.bottom_dock.isVisible()
        self.bottom_dock.setVisible(visible)
        self.action_toggle_bottom_dock.setText(
            "隐藏日志窗口" if visible else "显示日志窗口"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PackingSoftware()
    window.show()
    sys.exit(app.exec_())
