from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QGroupBox,
    QFormLayout,
    QTabWidget,
    QWidget,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap


class ReportDetailDialog(QDialog):
    def __init__(self, report, parent=None):
        super().__init__(parent)
        self.report = report
        self.setWindowTitle(f"装箱方案详情 - {report.solution_id}")
        self.setMinimumSize(800, 600)
        self.init_ui()

    def init_ui(self):
        # 主布局
        self.main_layout = QVBoxLayout(self)

        # 标签页
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # 基本信息标签页
        self.info_tab = QWidget()
        self.info_layout = QVBoxLayout(self.info_tab)
        self.create_info_tab()
        self.tabs.addTab(self.info_tab, "基本信息")

        # 装箱详情标签页
        self.detail_tab = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_tab)
        self.create_detail_tab()
        self.tabs.addTab(self.detail_tab, "装箱详情")

        # 统计图表标签页
        self.chart_tab = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_tab)
        self.create_chart_tab()
        self.tabs.addTab(self.chart_tab, "统计图表")

        # 按钮区域
        self.button_layout = QHBoxLayout()

        self.print_btn = QPushButton("打印")
        self.print_btn.clicked.connect(self.print_report)

        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)

        self.button_layout.addStretch()
        self.button_layout.addWidget(self.print_btn)
        self.button_layout.addWidget(self.close_btn)

        self.main_layout.addLayout(self.button_layout)

    def create_info_tab(self):
        """创建基本信息标签页"""
        # 基本信息组
        info_group = QGroupBox("方案信息")
        info_form = QFormLayout()

        info_form.addRow("方案ID:", QLabel(self.report.solution_id))
        info_form.addRow(
            "创建时间:", QLabel(self.report.creation_date.strftime("%Y-%m-%d %H:%M:%S"))
        )

        if self.report.container:
            container_info = f"{self.report.container.name} ({self.report.container.length}x{self.report.container.width}x{self.report.container.height}mm)"
            info_form.addRow("集装箱:", QLabel(container_info))

        info_form.addRow("总件数:", QLabel(str(self.report.total_items)))
        info_form.addRow("总体积(m³):", QLabel(f"{self.report.total_volume:.2f}"))
        info_form.addRow("总重量(kg):", QLabel(f"{self.report.total_weight:.2f}"))
        info_form.addRow("空间利用率:", QLabel(f"{self.report.utilization:.1%}"))
        info_form.addRow("稳定性:", QLabel(f"{self.report.stability:.1%}"))

        info_group.setLayout(info_form)
        self.info_layout.addWidget(info_group)

        # 缩略图
        if hasattr(self.report, "image_path") and self.report.image_path:
            thumb_group = QGroupBox("方案预览")
            thumb_layout = QVBoxLayout()

            pixmap = QPixmap(self.report.image_path)
            if not pixmap.isNull():
                label = QLabel()
                label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio))
                thumb_layout.addWidget(label)

            thumb_group.setLayout(thumb_layout)
            self.info_layout.addWidget(thumb_group)

        # 备注
        notes_group = QGroupBox("备注")
        notes_layout = QVBoxLayout()

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlainText(self.report.notes if self.report.notes else "")
        self.notes_edit.setReadOnly(True)

        notes_layout.addWidget(self.notes_edit)
        notes_group.setLayout(notes_layout)
        self.info_layout.addWidget(notes_group)

        self.info_layout.addStretch()

    def create_detail_tab(self):
        """创建装箱详情标签页"""
        # TODO: 实现装箱详情表格
        label = QLabel("装箱详情表格将在这里显示")
        label.setAlignment(Qt.AlignCenter)
        self.detail_layout.addWidget(label)
        self.detail_layout.addStretch()

    def create_chart_tab(self):
        """创建统计图表标签页"""
        # TODO: 实现统计图表
        label = QLabel("统计图表将在这里显示")
        label.setAlignment(Qt.AlignCenter)
        self.chart_layout.addWidget(label)
        self.chart_layout.addStretch()

    def print_report(self):
        """打印报表"""
        # TODO: 实现打印功能
        QtWidgets.QMessageBox.information(self, "提示", "打印功能将在后续版本实现")
