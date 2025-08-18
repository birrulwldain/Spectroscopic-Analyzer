from __future__ import annotations
import os
import sys

# Force pyqtgraph to use PySide6 (defensive)
os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PySide6')

# Allow running both `python -m app.main` and `python app/main.py`
if __package__ is None:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from PySide6 import QtWidgets as _QtWidgets  # type: ignore
QApplication = _QtWidgets.QApplication
from app.ui.main_window import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())