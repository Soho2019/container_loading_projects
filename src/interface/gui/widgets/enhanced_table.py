from PyQt5.QtWidgets import QTableWidget, QMenu, QAction
from PyQt5.QtCore import Qt


class EnhancedTableWidget(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_context_menu()

    def setup_context_menu(self):
        self.setContextMenuPolicy(Qt.ActionsContextMenu)

        # 添加右键菜单动作
        self.copy_action = QAction("复制", self)
        self.paste_action = QAction("粘贴", self)
        self.addAction(self.copy_action)
        self.addAction(self.paste_action)

        # 连接信号
        self.copy_action.triggered.connect(self.copy_selection)
        self.paste_action.triggered.connect(self.paste_to_cell)

    def copy_selection(self):
        # 实现复制功能
        pass

    def paste_to_cell(self):
        # 实现粘贴功能
        pass
