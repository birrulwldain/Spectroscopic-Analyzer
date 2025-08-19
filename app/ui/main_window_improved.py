from __future__ import annotations
from typing import TYPE_CHECKING, Any

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

# File utils dimasukkan langsung ke dalam kelas MainWindow

# Force pyqtgraph to use PySide6 to avoid mixing Qt bindings (defensive)
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
import pyqtgraph as pg

# Import langsung untuk runtime
from PySide6.QtWidgets import QDialog, QMessageBox, QTableWidgetItem

if TYPE_CHECKING:
    from PySide6.QtCore import QThread, Signal, Slot, Qt, QTimer
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QFileDialog,
        QLabel,
        QLineEdit,
        QGroupBox,
        QTableWidget,
        QHeaderView,
        QTabWidget,
        QFormLayout,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QSlider,
        QSplitter,
        QScrollArea,
        QSizePolicy,
    )
else:  # runtime imports
    from PySide6 import QtCore as _QtCore  # type: ignore
    from PySide6 import QtGui as _QtGui  # type: ignore
    from PySide6 import QtWidgets as _QtWidgets  # type: ignore

    QThread = _QtCore.QThread
    Signal = _QtCore.Signal
    Slot = _QtCore.Slot
    Qt = _QtCore.Qt
    QTimer = _QtCore.QTimer

    QColor = _QtGui.QColor

    QMainWindow = _QtWidgets.QMainWindow
    QWidget = _QtWidgets.QWidget
    QScrollArea = _QtWidgets.QScrollArea
    QSizePolicy = _QtWidgets.QSizePolicy
    QVBoxLayout = _QtWidgets.QVBoxLayout
    QHBoxLayout = _QtWidgets.QHBoxLayout
    QPushButton = _QtWidgets.QPushButton
    QFileDialog = _QtWidgets.QFileDialog
    QLabel = _QtWidgets.QLabel
    QLineEdit = _QtWidgets.QLineEdit
    QGroupBox = _QtWidgets.QGroupBox
    QTableWidget = _QtWidgets.QTableWidget
    QTableWidgetItem = _QtWidgets.QTableWidgetItem
    QHeaderView = _QtWidgets.QHeaderView
    QTabWidget = _QtWidgets.QTabWidget
    QFormLayout = _QtWidgets.QFormLayout
    QCheckBox = _QtWidgets.QCheckBox
    QComboBox = _QtWidgets.QComboBox
    QDoubleSpinBox = _QtWidgets.QDoubleSpinBox
    QSlider = _QtWidgets.QSlider
    QSplitter = _QtWidgets.QSplitter

from app.ui.worker import Worker
from app.ui.control_panel import ControlPanel
from app.ui.results_panel import ResultsPanel
from app.ui.batch_dialog import BatchDialog


class MainWindow(QMainWindow):
    analyzeRequested = Signal(dict)
    previewRequested = Signal(dict)

    worker_thread: "QThread"
    worker: "Worker"

    raw_asc_content: str | None
    last_results: dict | None

    current_wavelengths: np.ndarray | None
    current_intensities: np.ndarray | None
    current_peaks_wl: np.ndarray | None
    current_peaks_int: np.ndarray | None

    region: Any | None
    _region_proxy: Any | None

    overlays: list[Any]

    predict_tab_index: int
    validate_tab_index: int

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectroscopy AI Validator")
        self.setGeometry(100, 100, 1400, 900)  # Tingkatkan tinggi window untuk tampilan lebih baik

        # State
        # Data untuk manajemen multiple files
        self.asc_files = {}  # Dictionary untuk menyimpan file yang di-load
        self.current_file_name = None  # Nama file yang sedang aktif
        
        self.raw_asc_content: str | None = None
        self.last_results: dict | None = None
        self.current_wavelengths: np.ndarray | None = None
        self.current_intensities: np.ndarray | None = None
        self.current_peaks_wl: np.ndarray | None = None
        self.current_peaks_int: np.ndarray | None = None
        self.region = None
        self._region_proxy = None
        self._main_vline = None
        self._main_hline = None
        self._main_label = None
        self._zoom_vline = None
        self._zoom_hline = None
        self._zoom_label = None
        self._mouse_proxy_main = None
        self._mouse_proxy_zoom = None
        self.overlays: list[Any] = []

        # Debounce timer for interactive controls
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(400)
        self.debounce_timer.timeout.connect(self.trigger_interactive_analysis)

        # Background worker
        self.worker_thread = QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self.worker_thread)
        # Wire new preview/full API (connect finished after UI is built)
        self.previewRequested.connect(self.worker.run_preview)
        self.analyzeRequested.connect(self.worker.run_full)
        self.worker.error.connect(self.show_error)
        self.worker_thread.start()

        # Build UI
        self.setup_ui()
        # Now that UI is ready, wire worker results to ResultsPanel
        if hasattr(self, "results_panel"):
            self.worker.previewFinished.connect(self.results_panel.update_preview)
            self.worker.fullAnalysisFinished.connect(self.results_panel.update_full_results)
            # Also enable export button when full results include any table
            self.worker.fullAnalysisFinished.connect(self._on_full_finished_enable_export)
            # Keep MainWindow state (arrays for crosshair/overlay) in sync without replotting
            try:
                self.worker.previewFinished.connect(self._sync_state_from_results)
                self.worker.fullAnalysisFinished.connect(self._sync_state_from_results)
            except (TypeError, RuntimeError):
                pass

    def setup_ui(self):
        # Build modular panels
        left_panel = ControlPanel(self)
        results_panel = ResultsPanel(self)

        # Bind core widgets used by MainWindow logic
        self.file_button = left_panel.file_button
        self.status_label = left_panel.status_label
        self.tabs = left_panel.tabs
        # Buttons/signals are managed inside ControlPanel; keep references for compatibility
        self.gt_input = left_panel.gt_input
        self.predict_tab_index = left_panel.predict_tab_index
        self.validate_tab_index = left_panel.validate_tab_index
        self.prominence_slider = left_panel.prominence_slider
        self.prominence_label = left_panel.prominence_label
        self.distance_input = left_panel.distance_input
        self.height_input = left_panel.height_input
        self.width_input = left_panel.width_input
        self.threshold_input = left_panel.threshold_input
        self.baseline_switch = left_panel.baseline_switch
        self.baseline_overlay_switch = left_panel.baseline_overlay_switch
        self.raw_resolution_switch = left_panel.raw_resolution_switch
        self.abel_switch = left_panel.abel_switch
        self.lam_input = left_panel.lam_input
        self.p_input = left_panel.p_input
        self.niter_input = left_panel.niter_input
        self.smooth_switch = left_panel.smooth_switch
        self.sg_window_input = left_panel.sg_window_input
        self.sg_poly_input = left_panel.sg_poly_input
        self.norm_combo = left_panel.norm_combo
        self.shift_spin = left_panel.shift_spin
        self.overlay_button = left_panel.overlay_button
        self.export_button = left_panel.export_button
        self._left_panel_ref = left_panel  # keep reference for parameter access

        self.plot_widget = results_panel.plot_widget
        self.results_panel = results_panel

        # Wire signals
        self.file_button.clicked.connect(self.open_file_dialog)
        # Bridge ControlPanel signals to MainWindow signals/handlers
        left_panel.previewRequested.connect(
            lambda d: self.previewRequested.emit({**d, "asc_content": self.raw_asc_content})
        )
        # Connect regionChanged signal to update zoom plot
        def update_zoom(x0, x1):
            print(f"Signal handler called with {x0}, {x1}")
            self.results_panel.update_zoom_plot(self.zoom_plot_widget, x0, x1)
            
        # Add method to update sensitivity label
        def update_sensitivity_value(value):
            print(f"Updating sensitivity display: {value:.1f} nm")
            self.sensitivity_info.setText(f"Nilai sensitivitas: {value:.1f} nm")
            
        # Connect signals
        results_panel.regionChanged.connect(update_zoom)
        results_panel.sensitivityChanged.connect(update_sensitivity_value)
        left_panel.analysisRequested.connect(self._on_analysis_requested)
        self.prominence_slider.valueChanged.connect(self.on_slider_value_changed)
        self.export_button.clicked.connect(self.export_results_to_xlsx)
        self.overlay_button.clicked.connect(self.add_overlay_spectrum)
        left_panel.batch_button.clicked.connect(
            lambda: self.open_batch_dialog(self._left_panel_ref.get_parameters())
        )

        # Root layout
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create a splitter for the main sections (plots and controls)
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Upper section for plots (3/4 main plot + 1/4 zoom preview)
        plots_container = QWidget()
        plots_layout = QHBoxLayout(plots_container)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(10)  # Lebih besar untuk jarak yang lebih baik
        
        # Create a splitter for the plots section
        plots_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Buat widget untuk penampung plot
        plot_area = QWidget()
        # Use size policy instead of fixed height for better responsiveness
        plot_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Layout untuk area plot
        plot_layout = QVBoxLayout(plot_area)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        
        # Tambahkan widget plot with better size policy
        results_panel.plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Set a minimum height for the plot but allow it to grow
        results_panel.plot_widget.setMinimumHeight(300)
        plot_layout.addWidget(results_panel.plot_widget)
        
        # Buat widget untuk plot zoom
        zoom_area = QWidget()
        zoom_area.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        
        # Layout untuk area zoom
        zoom_layout = QVBoxLayout(zoom_area)
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        zoom_layout.setSpacing(5)
        
        # Label sensitivitas
        self.sensitivity_info = QLabel("Nilai sensitivitas: 5.0 nm")
        self.sensitivity_info.setStyleSheet("color: #0066cc; padding: 3px; font-weight: bold;")
        self.sensitivity_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zoom_layout.addWidget(self.sensitivity_info)
        
        # Create zoom plot widget with highly visible settings
        self.zoom_plot_widget = pg.PlotWidget()
        self.zoom_plot_widget.showGrid(x=True, y=True, alpha=0.5)
        self.zoom_plot_widget.setTitle("PREVIEW ZOOM")
        self.zoom_plot_widget.setBackground('lightblue')
        self.zoom_plot_widget.getViewBox().setBorder(pg.mkPen('b', width=2))
        
        # Use size policy for better responsiveness
        self.zoom_plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.zoom_plot_widget.setMinimumHeight(200)  # Minimum height but allow it to grow
        self.zoom_plot_widget.setMinimumWidth(200)
        
        zoom_layout.addWidget(self.zoom_plot_widget)
        
        # Add widgets to the plots splitter
        plots_splitter.addWidget(plot_area)
        plots_splitter.addWidget(zoom_area)
        
        # Set initial sizes - 75% for main plot, 25% for zoom
        plots_splitter.setSizes([75, 25])
        
        # Add splitter to plots container
        plots_layout.addWidget(plots_splitter)
        
        # Lower section for horizontal control panel with scroll area
        control_container = QWidget()
        control_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        control_layout = QHBoxLayout(control_container)
        control_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a splitter for file list and control panel
        control_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Buat file list widget untuk menampilkan daftar file yang dimuat
        self.file_list_widget = QWidget()
        self.file_list_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        file_list_layout = QVBoxLayout(self.file_list_widget)
        file_list_layout.setContentsMargins(5, 5, 5, 5)
        
        # Buat label untuk daftar file
        file_list_label = QLabel("Daftar File:")
        file_list_label.setStyleSheet("font-weight: bold; color: #0066cc; font-size: 11pt;")
        file_list_layout.addWidget(file_list_label)
        
        # Buat table widget untuk menampilkan daftar file
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(1)
        self.files_table.setHorizontalHeaderLabels(["Nama File"])
        self.files_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.files_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.files_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.files_table.setMinimumWidth(150)  # Reduced minimum width for better proportion
        self.files_table.setAlternatingRowColors(True)
        self.files_table.setStyleSheet("alternate-background-color: #f0f8ff;")
        self.files_table.verticalHeader().setVisible(False)  # Sembunyikan header nomor baris
        
        # Connect table selection ke fungsi untuk mengganti file
        self.files_table.itemSelectionChanged.connect(self.switch_selected_file)
        
        file_list_layout.addWidget(self.files_table)
        
        # Tambahkan label status jumlah file
        self.file_count_label = QLabel("0 file dimuat")
        self.file_count_label.setStyleSheet("color: #555; font-style: italic; font-size: 9pt;")
        self.file_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        file_list_layout.addWidget(self.file_count_label)
        
        # Tambahkan tombol buka file dan folder
        buttons_layout = QHBoxLayout()
        
        open_file_btn = QPushButton("Buka File...")
        open_file_btn.setMinimumHeight(30)
        open_file_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_file_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #004999;
            }
        """)
        open_file_btn.clicked.connect(self.open_file_dialog)
        buttons_layout.addWidget(open_file_btn)
        
        open_folder_btn = QPushButton("Buka Folder...")
        open_folder_btn.setMinimumHeight(30)
        open_folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #219653;
            }
        """)
        open_folder_btn.clicked.connect(self.open_folder_dialog)
        buttons_layout.addWidget(open_folder_btn)
        
        file_list_layout.addLayout(buttons_layout)
        
        # Create scroll area for control panel
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(left_panel)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Add widgets to control splitter
        control_splitter.addWidget(self.file_list_widget)
        control_splitter.addWidget(scroll_area)
        
        # Set initial sizes - 20% for file list, 80% for control panel
        control_splitter.setSizes([20, 80])
        
        # Add control splitter to control container
        control_layout.addWidget(control_splitter)
        
        # Add the plots container and control container to main splitter
        main_splitter.addWidget(plots_container)
        main_splitter.addWidget(control_container)
        
        # Set initial sizes for main sections - 40% for plots, 60% for controls
        main_splitter.setSizes([40, 60])
        
        # Add main splitter to root layout
        root_layout.addWidget(main_splitter)
        
        self.setCentralWidget(root)
        
        # Attach double-click reset on main view
        vb = self.plot_widget.getViewBox()
        if vb is not None:
            vb.mouseDoubleClickEvent = self.custom_mouse_double_click

        # Setup crosshair for both plots
        self._setup_crosshair(self.plot_widget, which="main")
        self._setup_crosshair(self.zoom_plot_widget, which="zoom")
        
        # Add initial dummy plot to zoom widget to ensure it's visible
        x = np.arange(100)
        y = np.sin(x/10) * 10
        self.zoom_plot_widget.plot(x, y, pen=pg.mkPen('b', width=2.5))
        
        # Setup a test data for zoom plot
        self.zoom_plot_widget.plot(np.arange(100), np.ones(100)*50, pen=pg.mkPen('r', width=3))
        
        # Need to monitor for when a file is loaded and then update the zoom
        self.previewRequested.connect(self._on_preview_requested_with_zoom)
        self.analyzeRequested.connect(self._on_analyze_requested_with_zoom)
        
        # Set initial sensitivity value
        if hasattr(self, 'sensitivity_info') and hasattr(results_panel, '_roi_shift_step'):
            self.sensitivity_info.setText(f"Nilai sensitivitas: {results_panel._roi_shift_step:.1f} nm")
