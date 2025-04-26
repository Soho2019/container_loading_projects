from PyQt5.QtCore import QFile, QTextStream


def load_stylesheet(widget, style_file="main_style.qss"):
    file = QFile(f":/resources/styles/{style_file}")
    if file.open(QFile.ReadOnly | QFile.Text):
        stream = QTextStream(file)
        widget.setStyleSheet(stream.readAll())
        file.close()
