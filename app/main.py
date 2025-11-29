"""
Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine

Interactive GUI application for spectroscopic data analysis using a deep learning model.

This implementation accompanies the paper:
    Walidain, B., Idris, N., Saddami, K., Yuzza, N., & Mitaphonna, R. (2025).
    "Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine."
    IOP Conference Series: Earth and Environmental Science, AIC 2025. (in press)

Authors:
    Birrul Walidain, Nasrullah Idris, Khairun Saddami, Natasya Yuzza, Rara Mitaphonna

For more information, see:
    - README.md: Installation and usage instructions
    - GitHub: https://github.com/birrulwaldain/informer-libs-aceh
"""
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