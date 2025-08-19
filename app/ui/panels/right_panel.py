from __future__ import annotations
from typing import TYPE_CHECKING

import pyqtgraph as pg

if TYPE_CHECKING:
    from PySide6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGroupBox,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QLabel,
        QPushButton,
        QCheckBox,
    )
else:
    from PySide6 import QtWidgets as _QtWidgets  # type: ignore

    QWidget = _QtWidgets.QWidget
    QVBoxLayout = _QtWidgets.QVBoxLayout
    QHBoxLayout = _QtWidgets.QHBoxLayout
    QGroupBox = _QtWidgets.QGroupBox
    QTableWidget = _QtWidgets.QTableWidget
    QTableWidgetItem = _QtWidgets.QTableWidgetItem
    QHeaderView = _QtWidgets.QHeaderView
    QLabel = _QtWidgets.QLabel
    QPushButton = _QtWidgets.QPushButton
    QCheckBox = _QtWidgets.QCheckBox


class RightPanel(QWidget):
    """Right-side plots and results table."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        plot_group = QGroupBox("Hasil Interpretasi Grafik")
        plot_layout = QVBoxLayout(plot_group)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setTitle("Plot Utama (Overview)")
        plot_layout.addWidget(self.plot_widget)

        self.zoom_plot_widget = pg.PlotWidget()
        self.zoom_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.zoom_plot_widget.setTitle("Plot Zoom (Area Terpilih)")
        plot_layout.addWidget(self.zoom_plot_widget)

        self.radial_plot_widget = pg.PlotWidget()
        self.radial_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.radial_plot_widget.setTitle("Profil Radial (Abel)")
        plot_layout.addWidget(self.radial_plot_widget)

        # Plot controls
        plot_controls_layout = QHBoxLayout()
        self.zoom_mode_checkbox = QCheckBox("Tampilkan Seleksi Zoom")
        self.zoom_mode_checkbox.setChecked(True)
        plot_controls_layout.addWidget(self.zoom_mode_checkbox)
        plot_controls_layout.addStretch()
        self.btn_reset = QPushButton("Reset View")
        plot_controls_layout.addWidget(self.btn_reset)
        plot_layout.addLayout(plot_controls_layout)

        table_group = QGroupBox("Tabel Hasil")
        table_layout = QVBoxLayout(table_group)
        self.table_widget = QTableWidget()
        table_layout.addWidget(self.table_widget)

        layout.addWidget(plot_group, 2)
        layout.addWidget(table_group, 1)

