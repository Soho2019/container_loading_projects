from PyQt5.QtGui import QIcon, QPixmap


def get_icon(name):
    return QIcon(f":/resources/icons/{name}")


def get_pixmap(name):
    return QPixmap(f":/resources/images/{name}")


def get_icon(name):
    """获取图标资源"""
    icon_map = {
        # ... 其他图标
        "view.png": ":/icons/view.png",
        "pdf.png": ":/icons/pdf.png",
        "excel.png": ":/icons/excel.png",
        "delete.png": ":/icons/delete.png",
        "compare.png": ":/icons/compare.png",
    }

    if name in icon_map:
        return QIcon(icon_map[name])
    return QIcon()
