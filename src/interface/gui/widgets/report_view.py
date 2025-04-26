import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QSplitter,
    QGroupBox,
    QTableWidgetItem,
    QPushButton,
    QFileDialog,
    QComboBox,
    QLineEdit,
    QDateEdit,
    QHeaderView,
)
from PyQt5.QtCore import Qt, QDate
from database.models import PackingReport, Container
from interface.gui.dialogs.report_dialog import ReportDetailDialog
from interface.gui.utils.resource_loader import get_icon


class ReportView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.session = parent.session if hasattr(parent, "session") else None
        self.init_ui()
        self.load_reports()

    def init_ui(self):
        # 主布局
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        # 搜索和过滤区域
        self.filter_group = QGroupBox("搜索与筛选")
        self.filter_layout = QHBoxLayout()

        # 搜索框
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索方案ID或备注...")
        self.search_input.textChanged.connect(self.load_reports)

        # 集装箱筛选
        self.container_filter = QComboBox()
        self.container_filter.addItem("所有集装箱", None)
        self.populate_container_filter()
        self.container_filter.currentIndexChanged.connect(self.load_reports)

        # 日期范围
        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDate(QDate.currentDate().addMonths(-1))
        self.date_from.dateChanged.connect(self.load_reports)

        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDate(QDate.currentDate())
        self.date_to.dateChanged.connect(self.load_reports)

        # 添加控件到筛选布局
        self.filter_layout.addWidget(QLabel("搜索:"))
        self.filter_layout.addWidget(self.search_input)
        self.filter_layout.addWidget(QLabel("集装箱:"))
        self.filter_layout.addWidget(self.container_filter)
        self.filter_layout.addWidget(QLabel("从:"))
        self.filter_layout.addWidget(self.date_from)
        self.filter_layout.addWidget(QLabel("到:"))
        self.filter_layout.addWidget(self.date_to)

        self.filter_group.setLayout(self.filter_layout)
        self.main_layout.addWidget(self.filter_group)

        # 报表表格
        self.report_table = QTableWidget()
        self.report_table.setColumnCount(8)
        self.report_table.setHorizontalHeaderLabels(
            [
                "方案ID",
                "集装箱",
                "日期",
                "总件数",
                "总体积(m³)",
                "总重量(kg)",
                "利用率",
                "稳定性",
            ]
        )
        self.report_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.report_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.report_table.doubleClicked.connect(self.view_report_detail)

        # 设置表头
        header = self.report_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)

        # 按钮区域
        self.button_layout = QHBoxLayout()

        self.view_btn = QPushButton("查看详情")
        self.view_btn.setIcon(get_icon("view.png"))
        self.view_btn.clicked.connect(self.view_report_detail)

        self.export_pdf_btn = QPushButton("导出PDF")
        self.export_pdf_btn.setIcon(get_icon("pdf.png"))
        self.export_pdf_btn.clicked.connect(self.export_to_pdf)

        self.export_excel_btn = QPushButton("导出Excel")
        self.export_excel_btn.setIcon(get_icon("excel.png"))
        self.export_excel_btn.clicked.connect(self.export_to_excel)

        self.delete_btn = QPushButton("删除")
        self.delete_btn.setIcon(get_icon("delete.png"))
        self.delete_btn.clicked.connect(self.delete_report)

        self.compare_btn = QPushButton("比较方案")
        self.compare_btn.setIcon(get_icon("compare.png"))
        self.compare_btn.clicked.connect(self.compare_reports)

        self.button_layout.addWidget(self.view_btn)
        self.button_layout.addWidget(self.export_pdf_btn)
        self.button_layout.addWidget(self.export_excel_btn)
        self.button_layout.addWidget(self.delete_btn)
        self.button_layout.addWidget(self.compare_btn)
        self.button_layout.addStretch()

        # 添加到主布局
        self.main_layout.addWidget(self.report_table)
        self.main_layout.addLayout(self.button_layout)

    def populate_container_filter(self):
        """填充集装箱筛选下拉框"""
        if not self.session:
            return

        containers = self.session.query(Container).all()
        for container in containers:
            self.container_filter.addItem(
                f"{container.name} ({container.length}x{container.width}x{container.height}mm)",
                container.container_id,
            )

    def load_reports(self):
        """加载报表数据"""
        if not self.session:
            return

        # 获取筛选条件
        search_text = self.search_input.text().strip()
        container_id = self.container_filter.currentData()
        date_from = self.date_from.date().toPyDate()
        date_to = self.date_to.date().toPyDate()

        # 构建查询
        query = self.session.query(PackingReport)

        if search_text:
            query = query.filter(
                (PackingReport.solution_id.contains(search_text))
                | (PackingReport.notes.contains(search_text))
            )

        if container_id:
            query = query.filter(PackingReport.container_id == container_id)

        query = query.filter(
            PackingReport.creation_date >= date_from,
            PackingReport.creation_date <= date_to,
        ).order_by(PackingReport.creation_date.desc())

        reports = query.all()

        # 填充表格
        self.report_table.setRowCount(0)
        for row_idx, report in enumerate(reports):
            self.report_table.insertRow(row_idx)

            # 方案ID
            self.report_table.setItem(row_idx, 0, QTableWidgetItem(report.solution_id))

            # 集装箱
            container_name = report.container.name if report.container else "未知"
            self.report_table.setItem(row_idx, 1, QTableWidgetItem(container_name))

            # 日期
            self.report_table.setItem(
                row_idx,
                2,
                QTableWidgetItem(report.creation_date.strftime("%Y-%m-%d %H:%M")),
            )

            # 总件数
            self.report_table.setItem(
                row_idx, 3, QTableWidgetItem(str(report.total_items))
            )

            # 总体积
            self.report_table.setItem(
                row_idx, 4, QTableWidgetItem(f"{report.total_volume:.2f}")
            )

            # 总重量
            self.report_table.setItem(
                row_idx, 5, QTableWidgetItem(f"{report.total_weight:.2f}")
            )

            # 利用率
            self.report_table.setItem(
                row_idx, 6, QTableWidgetItem(f"{report.utilization:.1%}")
            )

            # 稳定性
            self.report_table.setItem(
                row_idx, 7, QTableWidgetItem(f"{report.stability:.1%}")
            )

            # 存储整个报告对象
            self.report_table.item(row_idx, 0).setData(Qt.UserRole, report)

    def view_report_detail(self):
        """查看报表详情"""
        selected = self.report_table.selectionModel().selectedRows()
        if not selected:
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择一个报表")
            return

        row = selected[0].row()
        report = self.report_table.item(row, 0).data(Qt.UserRole)

        dialog = ReportDetailDialog(report, self)
        dialog.exec_()

    def export_to_pdf(self):
        """导出为PDF"""
        selected = self.report_table.selectionModel().selectedRows()
        if not selected:
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择一个报表")
            return

        row = selected[0].row()
        report = self.report_table.item(row, 0).data(Qt.UserRole)

        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出PDF",
            f"装箱方案_{report.solution_id}.pdf",
            "PDF文件 (*.pdf)",
            options=options,
        )

        if file_name:
            # TODO: 实现PDF导出逻辑
            QtWidgets.QMessageBox.information(
                self, "提示", "PDF导出功能将在后续版本实现"
            )

    def export_to_excel(self):
        """导出为Excel"""
        selected = self.report_table.selectionModel().selectedRows()
        if not selected:
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择一个报表")
            return

        row = selected[0].row()
        report = self.report_table.item(row, 0).data(Qt.UserRole)

        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出Excel",
            f"装箱方案_{report.solution_id}.xlsx",
            "Excel文件 (*.xlsx)",
            options=options,
        )

        if file_name:
            # TODO: 实现Excel导出逻辑
            QtWidgets.QMessageBox.information(
                self, "提示", "Excel导出功能将在后续版本实现"
            )

    def delete_report(self):
        """删除报表"""
        selected = self.report_table.selectionModel().selectedRows()
        if not selected:
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择一个报表")
            return

        row = selected[0].row()
        report = self.report_table.item(row, 0).data(Qt.UserRole)

        reply = QtWidgets.QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除方案 {report.solution_id} 吗?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )

        if reply == QtWidgets.QMessageBox.Yes:
            try:
                self.session.delete(report)
                self.session.commit()
                self.load_reports()
                QtWidgets.QMessageBox.information(self, "成功", "报表已删除")
            except Exception as e:
                self.session.rollback()
                QtWidgets.QMessageBox.critical(
                    self, "错误", f"删除报表时出错: {str(e)}"
                )

    def compare_reports(self):
        """比较两个方案"""
        selected = self.report_table.selectionModel().selectedRows()
        if len(selected) != 2:
            QtWidgets.QMessageBox.warning(self, "警告", "请选择两个报表进行比较")
            return

        report1 = self.report_table.item(selected[0].row(), 0).data(Qt.UserRole)
        report2 = self.report_table.item(selected[1].row(), 0).data(Qt.UserRole)

        # TODO: 实现比较功能
        QtWidgets.QMessageBox.information(self, "提示", "方案比较功能将在后续版本实现")
