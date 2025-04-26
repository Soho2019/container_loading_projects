import json
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QSpinBox,
    QDialogButtonBox,
)


class SettingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("系统设置")

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # 添加设置项
        self.animation_speed = QSpinBox()
        self.animation_speed.setRange(1, 10)
        form_layout.addRow("动画速度：", self.animation_speed)

        # 按钮盒
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addLayout(form_layout)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def save_settings(self):
        settings = {"animation_speed": self.animation_speed.value()}
        with open("settings.json", "w") as f:
            json.dump(settings, f)
