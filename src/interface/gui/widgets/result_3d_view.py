import sys
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
    QFormLayout,
)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph.opengl as gl
import numpy as np
from database.models import Container, Pallet, Product


class Result3DView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        self.current_solution = None
        self.highlighted_products = set()

    def init_ui(self):
        # 主布局使用水平分割器 (3D视图占2/3，信息面板占1/3)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(5)  # 设置分割线宽度

        # 左侧信息面板 (1/3宽度)
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(5, 5, 5, 5)  # 减小边距

        # 右侧3D视图区域 (2/3宽度)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)

        # 初始化3D视图
        self.init_3d_view()
        self.right_layout.addWidget(self.view3d, 1)

        # 初始化信息面板
        self.init_info_panel()

        # 将左右部件添加到分割器并设置比例
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setSizes(
            [self.width() // 3, self.width() * 2 // 3]
        )  # 设置初始比例 (1:2)

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.main_splitter)
        self.setLayout(main_layout)

        # 初始化旋转动画
        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self.rotate_view)
        self.rotation_angle = 0
        self.is_rotating = False

    def init_3d_view(self):
        """初始化3D视图"""
        self.view3d = gl.GLViewWidget()
        self.view3d.setCameraPosition(distance=5, elevation=30, azimuth=45)
        self.view3d.setBackgroundColor(QtGui.QColor(240, 240, 240))

        # 添加坐标轴
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(-0.5, 0, 0)
        self.view3d.addItem(gx)

        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -0.5, 0)
        self.view3d.addItem(gy)

        gz = gl.GLGridItem()
        gz.translate(0, 0, -0.5)
        self.view3d.addItem(gz)

    def init_info_panel(self):
        """初始化左侧信息面板"""
        # 1. 方案编号和统计信息
        self.solution_group = QGroupBox("装箱方案")
        self.solution_layout = QVBoxLayout()

        self.solution_id_label = QLabel("方案编号: 未指定")
        self.summary_label = QLabel("总件数: 0 | 总体积: 0.00m³ | 总重量: 0.00kg")
        self.solution_id_label.setStyleSheet("font-weight: bold;")
        self.summary_label.setStyleSheet("font-weight: bold;")
        self.solution_layout.addWidget(self.solution_id_label)
        self.solution_layout.addWidget(self.summary_label)
        self.solution_group.setLayout(self.solution_layout)
        self.left_layout.addWidget(self.solution_group)

        # 2. 货物列表表格
        self.product_group = QGroupBox("货物列表")
        self.product_layout = QVBoxLayout()

        self.product_table = QTableWidget()
        self.product_table.setColumnCount(6)
        self.product_table.setHorizontalHeaderLabels(
            ["品名", "尺寸(mm)", "重量(kg)", "码放方式", "是否承重", "总件数"]
        )

        # 设置表格样式
        self.product_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        self.product_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch
        )
        self.product_table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeToContents
        )
        self.product_table.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeToContents
        )

        self.product_table.verticalHeader().setVisible(False)  # 隐藏行号
        self.product_table.setShowGrid(False)  # 隐藏网格线
        self.product_table.setAlternatingRowColors(True)  # 交替行颜色

        # 紧凑样式
        self.product_table.setStyleSheet(
            """
            QTableWidget {
                font-size: 11px;
                border: none;
            }
            QHeaderView::section {
                padding: 2px;
                border: none;
                border-bottom: 1px solid #ddd;
            }
        """
        )

        self.product_table.itemSelectionChanged.connect(
            self.highlight_selected_products
        )
        self.product_layout.addWidget(self.product_table)
        self.product_group.setLayout(self.product_layout)
        self.left_layout.addWidget(self.product_group, 1)  # 占据剩余空间

        # 3. 装箱积载视图
        self.utilization_group = QGroupBox("装箱积载视图")
        self.utilization_layout = QFormLayout()

        self.utilization_label = QLabel("重量记录: 0% 0.00t | 空闲记录: 0% 0.00m³")
        self.utilization_layout.addRow(self.utilization_label)

        self.utilization_group.setLayout(self.utilization_layout)
        self.left_layout.addWidget(self.utilization_group)

        # 4. 重心偏移信息
        self.center_group = QGroupBox("重心偏移信息")
        self.center_layout = QFormLayout()

        self.center_range_label = QLabel("允许偏移: (±0,±0,0~0) | 实际偏移: (0.0,0.0)")
        self.center_layout.addRow(self.center_range_label)

        self.offset_label = QLabel("前后偏重: 0.00kg | 左右偏重: 0.00kg")
        self.center_layout.addRow(self.offset_label)

        self.center_group.setLayout(self.center_layout)
        self.left_layout.addWidget(self.center_group)

    def set_solution(self, solution):
        """设置要显示的装箱解决方案"""
        self.current_solution = solution
        self.render_solution()
        self.update_info_panel()

    def update_info_panel(self):
        """更新左侧信息面板"""
        if not self.current_solution:
            return

        # 1. 更新方案编号和统计信息
        solution_id = self.current_solution.get("solution_id", "未指定")
        total_items = sum(
            p.get("quantity", 1) for p in self.current_solution.get("products", [])
        )
        total_volume = sum(
            (
                p["dimensions"][0]
                * p["dimensions"][1]
                * p["dimensions"][2]
                * p.get("quantity", 1)
            )
            / 1e9
            for p in self.current_solution.get("products", [])
        )
        total_weight = sum(
            p.get("weight", 0) * p.get("quantity", 1)
            for p in self.current_solution.get("products", [])
        )

        self.solution_id_label.setText(f"方案编号: {solution_id}")
        self.summary_label.setText(
            f"总件数: {total_items} | 总体积: {total_volume:.2f}m³ | 总重量: {total_weight:.2f}kg"
        )

        # 2. 更新货物列表
        self.product_table.setRowCount(0)
        for idx, product in enumerate(self.current_solution.get("products", [])):
            self.product_table.insertRow(idx)

            # 品名
            name = product.get("name", product.get("sku", f"产品{idx}"))
            self.product_table.setItem(idx, 0, QTableWidgetItem(name))

            # 尺寸
            size = f"{product['dimensions'][0]}×{product['dimensions'][1]}×{product['dimensions'][2]}"
            self.product_table.setItem(idx, 1, QTableWidgetItem(size))

            # 重量
            weight = f"{product.get('weight', 0):.2f}"
            self.product_table.setItem(idx, 2, QTableWidgetItem(weight))

            # 码放方式
            placement = "允许旋转" if product.get("rotatable", True) else "固定方向"
            self.product_table.setItem(idx, 3, QTableWidgetItem(placement))

            # 是否承重
            load_bearing = "是" if product.get("load_bearing", False) else "否"
            self.product_table.setItem(idx, 4, QTableWidgetItem(load_bearing))

            # 总件数
            quantity = str(product.get("quantity", 1))
            self.product_table.setItem(idx, 5, QTableWidgetItem(quantity))

        # 3. 更新装箱积载视图
        container = self.current_solution.get("container")
        if container:
            container_volume = (
                container.length * container.width * container.height
            ) / 1e9
            utilization = (
                (total_volume / container_volume) * 100 if container_volume > 0 else 0
            )
            self.utilization_label.setText(
                f"重量记录: {utilization:.1f}% {total_weight/1000:.2f}t | "
                f"空闲记录: {100-utilization:.1f}% {container_volume-total_volume:.2f}m³"
            )

        # 4. 更新重心偏移信息
        center_offset = self.current_solution.get("center_offset", (0, 0, 0))
        allowed_offset = self.current_solution.get("allowed_offset", (12, 8, 129))
        front_back = self.current_solution.get("front_back_offset", 0)
        left_right = self.current_solution.get("left_right_offset", 0)

        self.center_range_label.setText(
            f"允许偏移: (±{allowed_offset[0]},±{allowed_offset[1]},0~{allowed_offset[2]}) | "
            f"实际偏移: ({center_offset[0]:.1f},{center_offset[1]:.1f},{center_offset[2]:.1f})"
        )
        self.offset_label.setText(
            f"前后偏重: {front_back:.2f}kg | 左右偏重: {left_right:.2f}kg"
        )

    def highlight_selected_products(self):
        """高亮显示选中的产品"""
        if not self.current_solution:
            return

        selected_rows = set(
            index.row() for index in self.product_table.selectedIndexes()
        )
        products = self.current_solution.get("products", [])

        # 清除之前的高亮
        for item in self.highlighted_products:
            if hasattr(item, "setColor"):
                item.setColor(None)  # 恢复默认颜色

        self.highlighted_products.clear()

        # 高亮选中的产品
        for row in selected_rows:
            if 0 <= row < len(products):
                product_sku = products[row].get("sku", "")
                # 在实际应用中，这里需要根据sku找到3D视图中的对应物品并高亮
                # 这里简化为打印日志
                print(f"高亮显示产品: {product_sku}")

    def render_solution(self):
        """渲染3D装箱结果"""
        if not self.current_solution:
            return

        # 清空当前视图
        self.clear_view()

        # 获取容器尺寸
        container = self.current_solution.get("container")
        if not container:
            return

        # 绘制集装箱
        self.draw_container(container)

        # 绘制托盘和货物
        pallets = self.current_solution.get("pallets", [])
        products = self.current_solution.get("products", [])

        for pallet in pallets:
            self.draw_pallet(pallet)

        for product in products:
            self.draw_product(product)

        # 自动调整视图
        self.auto_scale_view(container)

    def draw_container(self, container):
        """绘制集装箱"""
        # 集装箱尺寸（转换为米）
        length = container.length / 1000
        width = container.width / 1000
        height = container.height / 1000

        # 集装箱颜色
        color = (0.8, 0.8, 0.8, 0.5)  # 半透明灰色

        # 创建集装箱的线框
        verts = np.array(
            [
                [0, 0, 0],
                [length, 0, 0],
                [length, width, 0],
                [0, width, 0],
                [0, 0, height],
                [length, 0, height],
                [length, width, height],
                [0, width, height],
            ]
        )

        edges = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],  # 底面
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],  # 顶面
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],  # 侧面
            ]
        )

        container_mesh = gl.GLLinePlotItem(pos=verts, edges=edges, color=color, width=2)
        self.view3d.addItem(container_mesh)

    def draw_pallet(self, pallet):
        """绘制托盘"""
        # 托盘尺寸和位置（转换为米）
        x, y, z = pallet["position"]
        x /= 1000
        y /= 1000
        z /= 1000
        length = pallet["length"] / 1000
        width = pallet["width"] / 1000
        height = pallet["height"] / 1000

        # 托盘颜色
        color = (0.5, 0.3, 0.1, 0.8)  # 木质棕色

        # 创建托盘立方体
        verts = np.array(
            [
                [x, y, z],
                [x + length, y, z],
                [x + length, y + width, z],
                [x, y + width, z],
                [x, y, z + height],
                [x + length, y, z + height],
                [x + length, y + width, z + height],
                [x, y + width, z + height],
            ]
        )

        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # 底面
                [4, 5, 6],
                [4, 6, 7],  # 顶面
                [0, 1, 5],
                [0, 5, 4],  # 前面
                [1, 2, 6],
                [1, 6, 5],  # 右面
                [2, 3, 7],
                [2, 7, 6],  # 后面
                [3, 0, 4],
                [3, 4, 7],  # 左面
            ]
        )

        pallet_mesh = gl.GLMeshItem(
            vertexes=verts, faces=faces, color=color, smooth=False
        )
        self.view3d.addItem(pallet_mesh)

    def draw_product(self, product):
        """绘制货物"""
        # 货物尺寸和位置（转换为米）
        x, y, z = product["position"]
        x /= 1000
        y /= 1000
        z /= 1000
        length, width, height = product["dimensions"]
        length /= 1000
        width /= 1000
        height /= 1000

        # 根据货物类型选择颜色
        if product.get("fragile", 0) > 0:
            color = (1, 0, 0, 0.8)  # 红色表示易碎品
        else:
            color = (0, 0.5, 1, 0.8)  # 蓝色表示普通货物

        # 添加SKU标签
        sku = product.get("sku", "")
        if sku:
            text = gl.GLTextItem(
                pos=(x + length / 2, y + width / 2, z + height), text=sku
            )
            text.setColor(QtGui.QColor(0, 0, 0))
            text.setGLViewWidget(self.view3d)
            self.view3d.addItem(text)

        # 创建货物立方体
        verts = np.array(
            [
                [x, y, z],
                [x + length, y, z],
                [x + length, y + width, z],
                [x, y + width, z],
                [x, y, z + height],
                [x + length, y, z + height],
                [x + length, y + width, z + height],
                [x, y + width, z + height],
            ]
        )

        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # 底面
                [4, 5, 6],
                [4, 6, 7],  # 顶面
                [0, 1, 5],
                [0, 5, 4],  # 前面
                [1, 2, 6],
                [1, 6, 5],  # 右面
                [2, 3, 7],
                [2, 7, 6],  # 后面
                [3, 0, 4],
                [3, 4, 7],  # 左面
            ]
        )

        product_mesh = gl.GLMeshItem(
            vertexes=verts, faces=faces, color=color, smooth=False
        )
        self.view3d.addItem(product_mesh)

    def auto_scale_view(self, container):
        """自动调整视图以适应集装箱"""
        # 计算合适的观察距离
        max_dim = max(container.length, container.width, container.height) / 1000
        distance = max_dim * 2.5  # 2.5倍最大尺寸作为观察距离

        # 设置相机位置
        self.view3d.setCameraPosition(distance=distance, elevation=30, azimuth=45)

    def clear_view(self):
        """清除当前视图中的所有项目"""
        self.view3d.clear()

        # 重新添加坐标轴
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(-0.5, 0, 0)
        self.view3d.addItem(gx)

        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -0.5, 0)
        self.view3d.addItem(gy)

        gz = gl.GLGridItem()
        gz.translate(0, 0, -0.5)
        self.view3d.addItem(gz)

    def rotate_view(self):
        """旋转视图动画"""
        self.rotation_angle += 1
        if self.rotation_angle >= 360:
            self.rotation_angle = 0
        self.view3d.setCameraPosition(azimuth=self.rotation_angle)

    def toggle_rotation(self):
        """切换旋转动画"""
        # print(f"旋转状态切换: {not self.is_rotating} -> {self.is_rotating}")  # 调试
        self.is_rotating = not self.is_rotating
        if self.is_rotating:
            # print("启动旋转定时器")  # 调试
            self.rotation_timer.start(50)  # 每50ms更新一次
        else:
            # print("停止旋转定时器")  # 调试
            self.rotation_timer.stop()

    def zoom_in(self):
        """放大视图"""
        camera = self.view3d.cameraParams()
        self.view3d.setCameraPosition(distance=camera["distance"] * 0.9)

    def zoom_out(self):
        """缩小视图"""
        camera = self.view3d.cameraParams()
        self.view3d.setCameraPosition(distance=camera["distance"] * 1.1)

    def reset_view(self):
        """重置视图"""
        if self.current_solution and self.current_solution.get("container"):
            self.auto_scale_view(self.current_solution["container"])
        else:
            self.view3d.setCameraPosition(distance=5, elevation=30, azimuth=45)

    def export_image(self):
        """导出当前视图为图片"""
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存图片", "", "PNG图片 (*.png);;JPEG图片 (*.jpg)", options=options
        )

        if fileName:
            # 捕获当前视图
            img = self.view3d.grabFramebuffer()
            img.save(fileName)
