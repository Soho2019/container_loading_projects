import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QPushButton,
    QGroupBox,
    QFileDialog,
    QTextEdit,
)
from PyQt5.QtCore import Qt, QTimer, QSize
import pyqtgraph.opengl as gl
import numpy as np
import imageio
import tempfile
import os
from datetime import datetime


class SimulationView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("装箱过程动态仿真")  # 直接设置标题

        self.current_solution = None
        self.animation_items = []
        self.current_step = 0
        self.total_steps = 0
        self.animation_speed = 1  # 1x speed by default
        self.is_playing = False
        self.is_recording = False
        self.temp_images = []

        # 3D view items
        self.container_mesh = None
        self.pallet_meshes = []
        self.product_meshes = []

        self.init_ui()

    def init_ui(self):
        # 清除旧布局和控件
        self.clear_layout(self.layout())

        # 主布局使用水平分割器 (3D视图占2/3，控制面板占1/3)
        self.main_splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(5)

        # 左侧3D视图区域 (2/3宽度)
        self.view3d = gl.GLViewWidget()
        self.view3d.setCameraPosition(distance=5, elevation=30, azimuth=45)
        self.view3d.setBackgroundColor(QtGui.QColor(240, 240, 240))

        # 添加坐标轴
        self.add_axis()

        # 右侧控制面板 (1/3宽度)
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_layout.setContentsMargins(5, 5, 5, 5)
        self.control_layout.setSpacing(10)  # 设置控件之间的间距

        # 1. 仿真控制按钮组
        self.control_group = QGroupBox("仿真控制")
        self.control_group_layout = QVBoxLayout()

        # 播放控制按钮行
        self.playback_controls = QWidget()
        self.playback_layout = QHBoxLayout(self.playback_controls)
        self.playback_layout.setContentsMargins(0, 0, 0, 0)

        self.play_button = QPushButton()
        self.play_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        )
        self.play_button.setFixedSize(40, 40)
        self.play_button.clicked.connect(self.toggle_playback)

        self.pause_button = QPushButton()
        self.pause_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause)
        )
        self.pause_button.setFixedSize(40, 40)
        self.pause_button.clicked.connect(self.pause_playback)

        self.stop_button = QPushButton()
        self.stop_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop)
        )
        self.stop_button.setFixedSize(40, 40)
        self.stop_button.clicked.connect(self.stop_playback)

        self.playback_layout.addWidget(self.play_button)
        self.playback_layout.addWidget(self.pause_button)
        self.playback_layout.addWidget(self.stop_button)
        self.playback_layout.addStretch()

        # 进度条
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 100)
        self.progress_slider.valueChanged.connect(self.seek_animation)

        # 当前步骤显示
        self.step_label = QLabel("步骤: 0/0")

        self.control_group_layout.addWidget(self.playback_controls)
        self.control_group_layout.addWidget(self.progress_slider)
        self.control_group_layout.addWidget(self.step_label)
        self.control_group.setLayout(self.control_group_layout)

        # 2. 速度控制组
        self.speed_group = QGroupBox("速度控制")
        self.speed_layout = QHBoxLayout()

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 5)
        self.speed_slider.setValue(1)
        self.speed_slider.valueChanged.connect(self.set_animation_speed)

        self.speed_label = QLabel("速度: 1x")

        self.speed_layout.addWidget(self.speed_slider)
        self.speed_layout.addWidget(self.speed_label)
        self.speed_group.setLayout(self.speed_layout)

        # 3. 步骤描述区域
        self.step_description_group = QGroupBox("步骤描述")  # 添加标题
        self.step_description_layout = QVBoxLayout()

        self.step_description = QTextEdit()
        self.step_description.setReadOnly(True)
        self.step_description.setPlaceholderText("步骤执行描述将显示在这里...")
        self.step_description.setStyleSheet(
            "background-color: white; border: 1px solid #ccc;"
        )

        self.step_description_layout.addWidget(self.step_description)
        self.step_description_group.setLayout(self.step_description_layout)

        # 4. 导出控制组
        self.export_group = QGroupBox("导出选项")
        self.export_layout = QVBoxLayout()

        self.export_image_btn = QPushButton("截图当前视图")
        self.export_image_btn.clicked.connect(self.export_image)

        self.export_video_btn = QPushButton("导出仿真视频")
        self.export_video_btn.clicked.connect(self.export_video)

        self.export_layout.addWidget(self.export_image_btn)
        self.export_layout.addWidget(self.export_video_btn)
        self.export_group.setLayout(self.export_layout)

        # 添加到主布局
        self.control_layout.addWidget(self.control_group)
        self.control_layout.addWidget(self.speed_group)
        self.control_layout.addWidget(self.step_description_group)
        self.control_layout.addWidget(self.export_group)
        self.control_layout.addStretch()

        # 将左右部件添加到分割器
        self.main_splitter.addWidget(self.view3d)
        self.main_splitter.addWidget(self.control_panel)
        self.main_splitter.setSizes([self.width() * 2 // 3, self.width() // 3])

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.main_splitter)
        main_layout.setContentsMargins(0, 0, 0, 0)  # 移除主布局边距
        main_layout.setSpacing(0)  # 移除主布局间距
        self.setLayout(main_layout)

        # 初始化动画定时器
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.setInterval(1000 // 30)  # 30fps

        # 初始化录制定时器
        self.record_timer = QTimer()
        self.record_timer.timeout.connect(self.capture_frame)
        self.record_timer.setInterval(1000 // 30)  # 30fps

    def update_step_description(self, description):
        """更新步骤描述"""
        self.step_description.append(description)

    def update_animation(self):
        """更新动画到下一步"""
        if self.current_step >= self.total_steps:
            self.stop_playback()
            return

        step = self.animation_steps[self.current_step]

        if step["type"] == "pallet" and step["action"] == "add":
            self.draw_pallet(step["data"])
            self.update_step_description(f"放置托盘: {step['data']['position']}")
        elif step["type"] == "product" and step["action"] == "add":
            self.draw_product(step["data"])
            self.update_step_description(
                f"放置货物: {step['data']['sku']} 到位置 {step['data']['position']}"
            )

        self.current_step += 1
        self.progress_slider.setValue(self.current_step)
        self.update_step_label()

    def clear_layout(self, layout):
        """清除布局中的所有控件"""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    def add_axis(self):
        """添加坐标轴"""
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

    def set_solution(self, solution):
        """设置要仿真的装箱解决方案"""
        self.current_solution = solution
        self.prepare_animation()
        self.reset_animation()

        # 自动调整视图
        if self.current_solution and self.current_solution.get("container"):
            self.auto_scale_view(self.current_solution["container"])

    def prepare_animation(self):
        """准备动画数据"""
        if not self.current_solution:
            return

        # 清空当前视图
        self.clear_view()

        # 获取容器尺寸
        container = self.current_solution.get("container")
        if not container:
            return

        # 绘制集装箱(始终显示)
        self.draw_container(container)

        # 准备动画步骤
        self.animation_steps = []

        # 步骤1: 显示托盘
        pallets = self.current_solution.get("pallets", [])
        for pallet in pallets:
            self.animation_steps.append(
                {
                    "type": "pallet",
                    "data": pallet,
                    "action": "add",
                    "description": f"托盘 ({pallet['position']}) 放置到集装箱中",
                }
            )

        # 步骤2: 逐个添加货物
        products = self.current_solution.get("products", [])
        for product in products:
            self.animation_steps.append(
                {
                    "type": "product",
                    "data": product,
                    "action": "add",
                    "description": f"货物 ({product['sku']}) 放置到位置 ({product['position']})",
                }
            )

        self.total_steps = len(self.animation_steps)
        self.progress_slider.setMaximum(self.total_steps)
        self.update_step_label()
        self.step_description.clear()

    def reset_animation(self):
        """重置动画到开始状态"""
        self.current_step = 0
        self.is_playing = False
        self.animation_timer.stop()

        # 清空所有动态项目
        for item in self.animation_items:
            self.view3d.removeItem(item)
        self.animation_items = []
        self.pallet_meshes = []
        self.product_meshes = []

        # 只保留集装箱
        if self.container_mesh:
            self.view3d.addItem(self.container_mesh)

        self.update_step_label()
        self.progress_slider.setValue(0)
        self.step_description.clear()

    def update_animation(self):
        """更新动画到下一步"""
        if self.current_step >= self.total_steps:
            self.stop_playback()
            return

        step = self.animation_steps[self.current_step]

        if step["type"] == "pallet" and step["action"] == "add":
            self.draw_pallet(step["data"])
        elif step["type"] == "product" and step["action"] == "add":
            self.draw_product(step["data"])

        # 更新步骤描述
        self.update_step_description(step["description"])

        self.current_step += 1
        self.progress_slider.setValue(self.current_step)
        self.update_step_label()

        if self.current_step >= self.total_steps:
            self.stop_playback()

    def update_step_label(self):
        """更新步骤标签"""
        self.step_label.setText(f"步骤: {self.current_step}/{self.total_steps}")

    def toggle_playback(self):
        """切换播放状态"""
        if self.total_steps == 0:
            return

        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        """开始播放"""
        if self.current_step >= self.total_steps:
            self.reset_animation()

        self.is_playing = True
        self.animation_timer.start()
        self.play_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause)
        )

    def pause_playback(self):
        """暂停播放"""
        self.is_playing = False
        self.animation_timer.stop()
        self.play_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        )

    def stop_playback(self):
        """停止播放"""
        self.is_playing = False
        self.animation_timer.stop()
        self.play_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        )
        self.progress_slider.setValue(0)
        self.current_step = 0
        self.update_step_label()

    def seek_animation(self, value):
        """跳转到指定步骤"""
        if not self.is_playing and value != self.current_step:
            self.current_step = value
            self.reset_animation()

            # 执行到当前步骤
            for i in range(self.current_step):
                step = self.animation_steps[i]
                if step["type"] == "pallet" and step["action"] == "add":
                    self.draw_pallet(step["data"])
                elif step["type"] == "product" and step["action"] == "add":
                    self.draw_product(step["data"])

            self.update_step_label()

    def set_animation_speed(self, speed):
        """设置动画速度"""
        self.animation_speed = speed
        self.speed_label.setText(f"速度: {speed}x")
        self.animation_timer.setInterval(1000 // (30 * speed))

    def draw_container(self, container):
        """绘制集装箱"""
        # 清空现有集装箱
        if self.container_mesh:
            self.view3d.removeItem(self.container_mesh)

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

        self.container_mesh = gl.GLLinePlotItem(
            pos=verts, edges=edges, color=color, width=2
        )
        self.view3d.addItem(self.container_mesh)

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
        self.animation_items.append(pallet_mesh)
        self.pallet_meshes.append(pallet_mesh)

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
            self.animation_items.append(text)

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
        self.animation_items.append(product_mesh)
        self.product_meshes.append(product_mesh)

    def auto_scale_view(self, container):
        """自动调整视图以适应集装箱"""
        max_dim = max(container.length, container.width, container.height) / 1000
        distance = max_dim * 2.5  # 2.5倍最大尺寸作为观察距离
        self.view3d.setCameraPosition(distance=distance, elevation=30, azimuth=45)

    def clear_view(self):
        """清除当前视图中的所有项目"""
        self.view3d.clear()
        self.animation_items = []
        self.pallet_meshes = []
        self.product_meshes = []
        self.container_mesh = None
        self.add_axis()

    def export_image(self):
        """导出当前视图为图片"""
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存图片", "", "PNG图片 (*.png);;JPEG图片 (*.jpg)", options=options
        )

        if fileName:
            img = self.view3d.grabFramebuffer()
            img.save(fileName)

    def export_video(self):
        """导出仿真过程为视频"""
        if self.total_steps == 0:
            QtWidgets.QMessageBox.warning(self, "警告", "没有可导出的动画数据")
            return

        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存视频", "", "MP4视频 (*.mp4);;GIF动画 (*.gif)", options=options
        )

        if fileName:
            # 准备临时目录
            self.temp_images = []
            temp_dir = tempfile.mkdtemp()

            # 重置动画并开始录制
            self.reset_animation()
            self.is_recording = True
            self.record_timer.start()

            # 显示录制状态
            self.export_video_btn.setText("录制中...")
            self.export_video_btn.setEnabled(False)

    def capture_frame(self):
        """捕获当前帧用于视频录制"""
        if self.current_step >= self.total_steps:
            # 录制完成
            self.finish_recording()
            return

        # 执行下一步动画
        step = self.animation_steps[self.current_step]

        if step["type"] == "pallet" and step["action"] == "add":
            self.draw_pallet(step["data"])
        elif step["type"] == "product" and step["action"] == "add":
            self.draw_product(step["data"])

        self.current_step += 1

        # 捕获当前帧
        img = self.view3d.grabFramebuffer()
        temp_path = os.path.join(
            tempfile.gettempdir(), f"frame_{self.current_step:04d}.png"
        )
        img.save(temp_path)
        self.temp_images.append(temp_path)

    def finish_recording(self):
        """完成视频录制并保存"""
        self.record_timer.stop()
        self.is_recording = False

        try:
            # 创建视频文件
            with imageio.get_writer(self.video_file, fps=30) as writer:
                for image_path in self.temp_images:
                    image = imageio.imread(image_path)
                    writer.append_data(image)
                    os.remove(image_path)  # 删除临时文件

            QtWidgets.QMessageBox.information(self, "成功", "视频导出完成")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"视频导出失败: {str(e)}")

        finally:
            # 清理
            self.temp_images = []
            self.reset_animation()
            self.export_video_btn.setText("导出仿真视频")
            self.export_video_btn.setEnabled(True)
