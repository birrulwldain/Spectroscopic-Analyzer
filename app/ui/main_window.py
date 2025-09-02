from __future__ import annotations
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
        self.prominence_input = left_panel.prominence_input
        self.distance_input = left_panel.distance_input
        self.height_input = left_panel.height_input
        self.width_input = left_panel.width_input
        self.threshold_input = left_panel.threshold_input
        self.prediction_threshold_input = left_panel.prediction_threshold_input
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
        # prominence_input signal is handled in ControlPanel
        self.export_button.clicked.connect(self.export_results_to_xlsx)
        self.overlay_button.clicked.connect(self.add_overlay_spectrum)
        left_panel.batch_button.clicked.connect(
            lambda: self.open_batch_dialog(self._left_panel_ref.get_parameters())
        )

        # Root layout
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(5, 5, 5, 5)
        
        # Upper section for plots (3/4 main plot + 1/4 zoom preview)
        plots_container = QWidget()
        plots_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        plots_layout = QHBoxLayout(plots_container)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(10)  # Lebih besar untuk jarak yang lebih baik
        
        # Buat widget untuk penampung plot
        plot_area = QWidget()
        plot_area.setFixedHeight(320)  # Tinggi plot area ditambah sedikit
        plot_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        plots_layout.addWidget(plot_area, 75)  # 75% of width
        
        # Layout untuk area plot
        plot_layout = QVBoxLayout(plot_area)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        
        # Tambahkan widget plot
        results_panel.plot_widget.setMinimumHeight(300)  # Tinggi plot ditingkatkan
        results_panel.plot_widget.setMaximumHeight(300)  # Tinggi plot ditingkatkan
        plot_layout.addWidget(results_panel.plot_widget)
        plot_layout.addStretch()  # Tambahkan stretch untuk mengatur posisi plot di bagian atas
        
        # Buat widget untuk plot zoom dengan tinggi yang sama
        zoom_area = QWidget()
        zoom_area.setFixedHeight(320)  # Tinggi zoom area ditambah sedikit
        zoom_area.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        plots_layout.addWidget(zoom_area, 25)  # 25% of width
        
        # Layout untuk area zoom
        zoom_layout = QVBoxLayout(zoom_area)
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        zoom_layout.setSpacing(5)
        
        # Label sensitivitas
        self.sensitivity_info = QLabel("Nilai sensitivitas: 5.0 nm")
        self.sensitivity_info.setStyleSheet("color: #0066cc; padding: 3px; font-weight: bold;")
        self.sensitivity_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sensitivity_info.setFixedHeight(20)  # Tinggi label dikurangi sedikit
        zoom_layout.addWidget(self.sensitivity_info)
        
        # Create zoom plot widget with highly visible settings
        self.zoom_plot_widget = pg.PlotWidget()
        self.zoom_plot_widget.showGrid(x=True, y=True, alpha=0.5)
        self.zoom_plot_widget.setTitle("PREVIEW ZOOM")
        self.zoom_plot_widget.setBackground('lightblue')
        self.zoom_plot_widget.getViewBox().setBorder(pg.mkPen('b', width=2))
        
        # Tetapkan tinggi zoom plot sejajar dengan plot utama
        self.zoom_plot_widget.setMinimumHeight(280)  # Tinggi zoom plot ditingkatkan
        self.zoom_plot_widget.setMaximumHeight(280)  # Tinggi zoom plot ditingkatkan
        self.zoom_plot_widget.setMinimumWidth(250)
        
        zoom_layout.addWidget(self.zoom_plot_widget)
        zoom_layout.addStretch()  # Tambahkan stretch untuk mengatur posisi zoom plot di bagian atas
        
        # Add plots section to root layout
        root_layout.addWidget(plots_container)  # Biarkan tinggi ditentukan oleh ukuran fixed dari widget-widget di dalamnya
        
        # Lower section for horizontal control panel with scroll area
        control_container = QWidget()
        control_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        control_layout = QHBoxLayout(control_container)
        control_layout.setContentsMargins(0, 0, 0, 0)
        
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
        self.files_table.setMinimumWidth(180)
        self.files_table.setMaximumWidth(250)
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
        
        # Tambahkan tombol ekspor publikasi
        try:
            import matplotlib.pyplot as plt
            MATPLOTLIB_AVAILABLE = True
            print("DEBUG: Matplotlib tersedia, membuat tombol ekspor publikasi")
        except ImportError:
            MATPLOTLIB_AVAILABLE = False
            print("DEBUG: Matplotlib tidak tersedia")
            
        if MATPLOTLIB_AVAILABLE:
            export_pub_btn = QPushButton("📊 Export Publication Plot")
            export_pub_btn.setMinimumHeight(35)
            export_pub_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            export_pub_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border-radius: 4px;
                    padding: 8px;
                    font-weight: bold;
                    font-size: 11px;
                    border: 2px solid #c0392b;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                    border: 2px solid #a93226;
                }
                QPushButton:pressed {
                    background-color: #a93226;
                }
                QPushButton:disabled {
                    background-color: #95a5a6;
                    color: #7f8c8d;
                    border: 2px solid #7f8c8d;
                }
            """)
            export_pub_btn.setToolTip("Export current zoom region as publication-ready plot\n(PNG, PDF, SVG, EPS formats)")
            export_pub_btn.clicked.connect(self._export_publication_plot)
            buttons_layout.addWidget(export_pub_btn)
            self.export_pub_button = export_pub_btn
            
            # Add ASC data export button
            export_asc_btn = QPushButton("📄 Export ASC Data")
            export_asc_btn.setMinimumHeight(35)
            export_asc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            export_asc_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2E8B57;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: #3CB371;
                    transform: translateY(-1px);
                }
                QPushButton:pressed {
                    background-color: #228B22;
                    transform: translateY(0px);
                }
                QPushButton:disabled {
                    background-color: #CCCCCC;
                    color: #666666;
                }
            """)
            export_asc_btn.setToolTip("Export current spectrum data as ASC file\n(Wavelength and Intensity data in tab-separated format)")
            export_asc_btn.clicked.connect(self._export_asc_data)
            buttons_layout.addWidget(export_asc_btn)
            self.export_asc_button = export_asc_btn
            print("DEBUG: Tombol ekspor publikasi ditambahkan ke file list area")

        file_list_layout.addLayout(buttons_layout)
        
        # Tambahkan stretching di bagian bawah file list layout
        file_list_layout.addStretch()
        
        # Buat scroll area untuk menampung control panel
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Penting! Memastikan widget di dalamnya dapat diubah ukurannya
        scroll_area.setWidget(left_panel)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Pastikan scroll area memiliki ukuran yang cukup
        scroll_area.setMinimumHeight(450)  # Tinggi minimum agar lebih banyak kontrol terlihat
        
        # Tambahkan daftar file dan control panel ke layout utama
        control_layout.addWidget(self.file_list_widget, 15)  # 15% lebar untuk daftar file
        control_layout.addWidget(scroll_area, 85)  # 85% lebar untuk panel kontrol
        
        # Add control panel to root layout dengan proporsi yang lebih besar
        root_layout.addWidget(control_container, 60)  # 60% of height untuk panel kontrol
        
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

        # Attach double-click reset on main view
        vb = self.plot_widget.getViewBox()
        if vb is not None:
            vb.mouseDoubleClickEvent = self.custom_mouse_double_click
            
        # Initialize ASC export button state
        self._update_asc_export_button()

    def open_batch_dialog(self, params_template: dict[str, Any]):
        dlg = BatchDialog(self, params_template)
        try:
            dlg.batchSummaryReady.connect(self._on_batch_summary_ready)  # type: ignore[arg-type]
        except (RuntimeError, TypeError):
            pass
        dlg.show()

    def _on_batch_summary_ready(self, df_obj: Any, saved_path: str):
        # Update status and render DataFrame into the table view
        self.status_label.setText(f"Batch selesai. Ringkasan: {saved_path}")
        df = None
        if isinstance(df_obj, pd.DataFrame):
            df = df_obj
        elif isinstance(df_obj, (list, dict)):
            try:
                df = pd.DataFrame(df_obj)
            except (ValueError, TypeError):
                df = None
        # Removed table display logic as `table_widget` no longer exists
        if df is None or getattr(df, 'empty', False):
            return
        headers = list(map(str, df.columns.tolist()))
        for i, (_, row) in enumerate(df.iterrows()):
            for j, key in enumerate(headers):
                pass  # Placeholder for future logic if needed

    def _on_analysis_requested(self, data: dict):
        # Ensure file content is included and forward to worker
        payload = {**data, "asc_content": self.raw_asc_content}
        self.analyzeRequested.emit(payload)

    def parse_asc_content(self, text: str) -> np.ndarray:
        rows: list[tuple[float, float]] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith(("#", "//", "%", ";")):
                continue
            line = line.replace(",", " ")
            parts = [p for p in line.split() if p]
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                rows.append((x, y))
            except ValueError:
                continue
        if not rows:
            raise ValueError("Tidak menemukan pasangan data numerik (x y).")
        return np.asarray(rows, dtype=float)
        
    def _update_file_list_display(self):
        """
        Update tampilan daftar file di table widget
        """
        if not hasattr(self, 'files_table') or not hasattr(self, 'asc_files'):
            return
        
        # Hapus semua baris yang ada
        self.files_table.setRowCount(0)
        
        # Tambahkan file ke tabel
        if not self.asc_files:
            # Update status label jika ada
            if hasattr(self, 'status_label'):
                self.status_label.setText("Tidak ada file yang dimuat")
            # Update file count label
            if hasattr(self, 'file_count_label'):
                self.file_count_label.setText("0 file dimuat")
            return
        
        # Update status label jika ada
        file_count = len(self.asc_files)
        if hasattr(self, 'status_label'):
            self.status_label.setText(f"{file_count} file{'s' if file_count > 1 else ''} dimuat")
        # Update file count label
        if hasattr(self, 'file_count_label'):
            if file_count == 1:
                self.file_count_label.setText("1 file dimuat")
            else:
                self.file_count_label.setText(f"{file_count} file dimuat")
        
        # Atur jumlah baris sesuai dengan jumlah file
        self.files_table.setRowCount(len(self.asc_files))
        
        # Sortir berdasarkan nama file
        sorted_files = sorted(self.asc_files.items())
        
        # Isi tabel dengan nama file yang sudah diurutkan
        for i, (filename, file_data) in enumerate(sorted_files):
            item = QTableWidgetItem(filename)
            
            # Tambahkan tooltip dengan informasi file
            try:
                file_size = os.path.getsize(file_data["path"])
                if file_size < 1024:
                    size_str = f"{file_size} B"
                elif file_size < 1024*1024:
                    size_str = f"{file_size/1024:.1f} KB"
                else:
                    size_str = f"{file_size/(1024*1024):.1f} MB"
                
                item.setToolTip(f"Nama: {filename}\nUkuran: {size_str}\nPath: {file_data['path']}")
            except:
                item.setToolTip(f"Nama: {filename}\nPath: {file_data['path']}")
            
            self.files_table.setItem(i, 0, item)
            
            # Jika file ini adalah file yang aktif, pilih barisnya dan highlight
            if self.current_file_name == filename:
                self.files_table.selectRow(i)
                item.setBackground(QColor(173, 216, 230))  # Light blue
                font = item.font()
                font.setBold(True)
                item.setFont(font)

    def _export_publication_plot(self):
        """Export current zoom region as publication-ready plot"""
        try:
            # Check if results panel has export method
            if hasattr(self.results_panel, 'export_publication_plot'):
                self.results_panel.export_publication_plot()
            else:
                QMessageBox.warning(self, "Export Error", 
                                  "Export functionality not available in results panel")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                               f"Failed to export plot: {str(e)}")

    def _export_asc_data(self):
        """Export current spectrum data as ASC file"""
        if self.current_wavelengths is None or self.current_intensities is None:
            QMessageBox.warning(self, "Export Error", 
                              "No spectrum data available for export")
            return
            
        if not self.current_file_name:
            QMessageBox.warning(self, "Export Error", 
                              "No file selected for export")
            return
            
        # Get filename for export
        base_name = os.path.splitext(self.current_file_name)[0]
        default_filename = f"{base_name}_exported.asc"
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Spectrum Data as ASC",
            default_filename,
            "ASC files (*.asc);;Text files (*.txt);;All files (*.*)"
        )
        
        if not filename:
            return
            
        try:
            # Write ASC format: tab-separated wavelength and intensity
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# Wavelength (nm)\tIntensity\n")
                for wl, intensity in zip(self.current_wavelengths, self.current_intensities):
                    f.write(f"{wl:.5f}\t{intensity:.1f}\n")
            
            self.status_label.setText(f"ASC data exported to: {filename}")
            QMessageBox.information(self, "Export Successful", 
                                  f"Spectrum data exported successfully to:\n{filename}")
                                  
        except Exception as e:
            self.status_label.setText(f"Failed to export ASC data: {str(e)}")
            QMessageBox.critical(self, "Export Error", 
                               f"Failed to export ASC data:\n{str(e)}")

    def _update_asc_export_button(self):
        """Enable/disable ASC export button based on data availability"""
        if hasattr(self, 'export_asc_button'):
            has_data = (self.current_wavelengths is not None and 
                       self.current_intensities is not None)
            self.export_asc_button.setEnabled(has_data)

    def open_file_dialog(self):
        # Dialog untuk memilih banyak file sekaligus
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "Buka File ASC (bisa pilih banyak)", "", "ASC Files (*.asc);;All Files (*)"
        )
        if not filenames:
            return
        
        # Proses file yang dipilih
        self._process_selected_files(filenames)
    
    def open_folder_dialog(self):
        # Dialog untuk memilih folder berisi file ASC
        folder_path = QFileDialog.getExistingDirectory(
            self, "Pilih Folder Berisi File ASC"
        )
        
        if not folder_path:
            return
        
        # Dialog untuk memilih opsi scanning subfolder
        include_subfolders = False
        result = QMessageBox.question(
            self,
            "Opsi Folder",
            "Apakah Anda ingin mencari file ASC di subfolder juga?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if result == QMessageBox.StandardButton.Yes:
            include_subfolders = True
        
        # Cari semua file ASC di dalam folder
        asc_files = []
        
        if include_subfolders:
            # Scan rekursif ke dalam semua subfolder
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(".asc"):
                        asc_files.append(os.path.join(root, file))
        else:
            # Hanya scan folder utama saja
            for file in os.listdir(folder_path):
                if file.lower().endswith(".asc"):
                    asc_files.append(os.path.join(folder_path, file))
        
        if not asc_files:
            QMessageBox.information(self, "Informasi", "Tidak ada file ASC yang ditemukan di folder ini.")
            return
        
        # Tampilkan pesan jika banyak file ditemukan
        if len(asc_files) > 10:
            result = QMessageBox.question(
                self, 
                "Konfirmasi", 
                f"Ditemukan {len(asc_files)} file ASC di folder{' dan subfolder' if include_subfolders else ''}. Lanjutkan memuat semua file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if result == QMessageBox.StandardButton.No:
                return
            
        # Proses file-file ASC yang ditemukan
        self._process_selected_files(asc_files)
    
    def _process_selected_files(self, filenames):
        # Inisialisasi dictionary untuk menyimpan konten file
        self.asc_files = {}
        
        # Baca semua file yang dipilih
        for filename in filenames:
            encodings = ["utf-8", "utf-16", "latin-1"]
            last_err = None
            content = None
            
            # Coba baca dengan berbagai encoding
            for enc in encodings:
                try:
                    with open(filename, "r", encoding=enc, errors="strict") as f:
                        content = f.read()
                    break
                except (UnicodeDecodeError, OSError) as e:
                    last_err = e
            
            # Jika berhasil baca file, simpan dalam dictionary
            if content is not None:
                base = os.path.basename(filename)
                self.asc_files[base] = {
                    "content": content,
                    "path": filename
                }
        
        # Jika tidak ada file yang berhasil dibaca
        if not self.asc_files:
            self.status_label.setText("Gagal membaca file-file yang dipilih")
            return
        
        # Perbarui tampilan daftar file
        self._update_file_list_display()
        
        # Tampilkan dialog pemilihan file untuk diproses
        self.show_file_selection_dialog()
    
    def show_file_selection_dialog(self):
        """Tampilkan dialog untuk memilih file yang akan diproses"""
        if not hasattr(self, 'asc_files') or not self.asc_files:
            return
            
        # Buat dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Pilih File untuk Diproses")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(500)
        
        # Layout utama
        layout = QVBoxLayout(dialog)
        
        # Label instruksi
        label = QLabel(f"<b>Ditemukan {len(self.asc_files)} file ASC. Pilih file yang ingin diproses:</b>")
        layout.addWidget(label)
        
        # Tambahkan list widget untuk menampilkan file
        list_widget = QTableWidget()
        list_widget.setColumnCount(3)  # Nama File, Ukuran, Path
        list_widget.setHorizontalHeaderLabels(["Nama File", "Ukuran", "Lokasi File"])
        list_widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        list_widget.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        list_widget.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        list_widget.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        list_widget.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        list_widget.setAlternatingRowColors(True)
        list_widget.setStyleSheet("alternate-background-color: #f0f8ff;")
        
        # Sortir daftar file berdasarkan nama file
        sorted_files = sorted(self.asc_files.items())
        
        # Isi list widget dengan file yang telah dibaca dan diurutkan
        list_widget.setRowCount(len(sorted_files))
        for i, (name, data) in enumerate(sorted_files):
            list_widget.setItem(i, 0, QTableWidgetItem(name))
            
            # Tambahkan informasi ukuran file
            try:
                file_size = os.path.getsize(data["path"])
                if file_size < 1024:
                    size_str = f"{file_size} B"
                elif file_size < 1024*1024:
                    size_str = f"{file_size/1024:.1f} KB"
                else:
                    size_str = f"{file_size/(1024*1024):.1f} MB"
                list_widget.setItem(i, 1, QTableWidgetItem(size_str))
            except:
                list_widget.setItem(i, 1, QTableWidgetItem("N/A"))
                
            list_widget.setItem(i, 2, QTableWidgetItem(data["path"]))
        
        layout.addWidget(list_widget)
        
        # Tambahkan informasi
        info_label = QLabel("<i>Klik dua kali pada file untuk langsung memuat, atau gunakan tombol di bawah.</i>")
        layout.addWidget(info_label)
        
        # Connect double-click untuk langsung memuat file
        list_widget.itemDoubleClicked.connect(lambda: self.load_selected_file(list_widget, dialog))
        
        # Tombol-tombol
        button_layout = QHBoxLayout()
        load_button = QPushButton("Muat File Terpilih")
        load_button.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #004999;
            }
        """)
        cancel_button = QPushButton("Batal")
        button_layout.addWidget(load_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        # Connect tombol ke fungsi
        cancel_button.clicked.connect(dialog.reject)
        load_button.clicked.connect(lambda: self.load_selected_file(list_widget, dialog))
        
        # Tampilkan dialog
        dialog.exec()
    
    def switch_selected_file(self):
        """Mengganti file yang aktif berdasarkan pilihan di table"""
        selected_items = self.files_table.selectedItems()
        if not selected_items:
            return
            
        # Ambil nama file yang dipilih
        filename = selected_items[0].text()
        if filename not in self.asc_files:
            return
        
        # Simpan nama file yang aktif
        self.current_file_name = filename
        
        # Muat konten file yang dipilih
        content = self.asc_files[filename]["content"]
        if content is None:
            self.status_label.setText(f"Konten file {filename} kosong")
            return
            
        # Set raw_asc_content dengan konten file yang dipilih
        self.raw_asc_content = content
        self.status_label.setText(f"File: {filename}")
        self.last_results = None
        self.export_button.setEnabled(False)
        
        # Pastikan raw_asc_content bukan None sebelum memproses
        if self.raw_asc_content is None:
            self.status_label.setText("Error: Konten file tidak valid")
            return
            
        try:
            # Cast ke str untuk memastikan type checker puas
            content_str: str = self.raw_asc_content
            data = self.parse_asc_content(content_str)
        except (ValueError, IndexError) as e:
            self.plot_widget.clear()
            self.status_label.setText(f"Format file tidak valid: {e}")
            return
            
        self.current_wavelengths = data[:, 0]
        self.current_intensities = data[:, 1]
        self.current_peaks_wl = None
        self.current_peaks_int = None
        
        # Update ASC export button state
        self._update_asc_export_button()
        
        # Trigger a quick preview so ResultsPanel draws the loaded signal
        self.previewRequested.emit(self.get_input_data())
    
    def load_selected_file(self, list_widget, dialog):
        """Muat file yang dipilih dari list widget"""
        selected_rows = list_widget.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(dialog, "Peringatan", "Silakan pilih file terlebih dahulu!")
            return
            
        # Ambil nama file yang dipilih
        row = selected_rows[0].row()
        filename = list_widget.item(row, 0).text()
        
        # Muat konten file yang dipilih
        content = self.asc_files[filename]["content"]
        if content is None:
            self.status_label.setText(f"Konten file {filename} kosong")
            dialog.accept()
            return
            
        # Set raw_asc_content dengan konten file yang dipilih
        self.raw_asc_content = content
        self.status_label.setText(f"File: {filename}")
        self.last_results = None
        self.export_button.setEnabled(False)
        
        # Pastikan raw_asc_content bukan None sebelum memproses
        if self.raw_asc_content is None:
            self.status_label.setText("Error: Konten file tidak valid")
            dialog.accept()
            return
            
        try:
            # Cast ke str untuk memastikan type checker puas
            content_str: str = self.raw_asc_content
            data = self.parse_asc_content(content_str)
        except (ValueError, IndexError) as e:
            self.plot_widget.clear()
            self.status_label.setText(f"Format file tidak valid: {e}")
            dialog.accept()  # Tutup dialog meskipun format tidak valid
            return
            
        self.current_wavelengths = data[:, 0]
        self.current_intensities = data[:, 1]
        self.current_peaks_wl = None
        self.current_peaks_int = None
        
        # Update ASC export button state
        self._update_asc_export_button()
        
        # Trigger a quick preview so ResultsPanel draws the loaded signal
        self.previewRequested.emit(self.get_input_data())
        
        # Tutup dialog
        dialog.accept()
        
        # Update daftar file dan tampilan
        self.current_file_name = filename
        self._update_file_list_display()

    def get_input_data(self) -> dict[str, Any]:
        def to_float(s: str | None):
            return float(s) if s else None

        def to_int(s: str | None):
            return int(s) if s else None

        try:
            prominence_val = float(self.prominence_input.text())
        except ValueError:
            prominence_val = 0.01  # default value
        return {
            "asc_content": self.raw_asc_content,
            "prominence": prominence_val,
            "distance": to_int(self.distance_input.text()),
            "height": to_float(self.height_input.text()),
            "width": to_float(self.width_input.text()),
            "threshold": to_float(self.threshold_input.text()),
            "use_raw_resolution": self.raw_resolution_switch.isChecked(),
            "apply_baseline_correction": self.baseline_switch.isChecked(),
            "show_baseline_overlay": self.baseline_overlay_switch.isChecked(),
            "compute_abel": self.abel_switch.isChecked(),
            "lam": to_float(self.lam_input.text()),
            "p": to_float(self.p_input.text()),
            "niter": to_int(self.niter_input.text()),
            "smoothing": self.smooth_switch.isChecked(),
            "sg_window": to_int(self.sg_window_input.text()),
            "sg_poly": to_int(self.sg_poly_input.text()),
            "normalization": self.norm_combo.currentText(),
            "shift_nm": float(self.shift_spin.value()),
        }

    def start_prediction(self):
        if not self.raw_asc_content:
            return
        data = self.get_input_data()
        data["analysis_mode"] = "predict"
        self.analyzeRequested.emit(data)

    def start_preprocess(self):
        if not self.raw_asc_content:
            return
        # Quick preview: only preprocessing results
        data = self.get_input_data()
        data["analysis_mode"] = "preprocess"
        self.previewRequested.emit(data)

    def start_validation(self):
        if not self.raw_asc_content:
            return
        input_data = self.get_input_data()
        input_data["analysis_mode"] = "validate"
        input_data["ground_truth_elements"] = [
            el.strip() for el in self.gt_input.text().split(",") if el.strip()
        ]
        self.analyzeRequested.emit(input_data)

    def export_results_to_xlsx(self):
        if not self.last_results:
            self.status_label.setText("Tidak ada data untuk diekspor.")
            return
        table_data = (
            self.last_results.get("validation_table")
            if self.last_results.get("validation_table")
            else self.last_results.get("prediction_table", [])
        )
        if not table_data:
            self.status_label.setText("Tidak ada data tabel untuk diekspor.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Simpan Hasil Excel", "", "Excel Files (*.xlsx)"
        )
        if not filename:
            return
        try:
            pd.DataFrame(table_data).to_excel(
                filename, index=False, sheet_name="Hasil Analisis"
            )
            self.status_label.setText(f"Hasil berhasil disimpan di {filename}")
        except (OSError, ValueError) as e:
            self.status_label.setText(f"Gagal menyimpan file: {e}")

    @Slot(dict)
    def update_results(self, results: dict[str, Any]):
        self.last_results = results
        self.export_button.setEnabled(
            bool(results.get("prediction_table") or results.get("validation_table"))
        )
        self.current_wavelengths = results["wavelengths"]
        self.current_intensities = results["spectrum_data"]
        self.current_peaks_wl = None
        self.current_peaks_int = None

        self.plot_widget.clear()
        self.overlays.clear()

        mode = (results.get("analysis_mode") or "predict").lower()
        self.plot_widget.setTitle(
            "Hasil Preprocessing" if mode == "preprocess" else "Hasil Analisis"
        )
        self.plot_widget.plot(
            self.current_wavelengths,
            self.current_intensities,
            pen=pg.mkPen("b", width=1.5),
        )
        if self.baseline_overlay_switch.isChecked() and results.get("baseline") is not None:
            try:
                self.plot_widget.plot(
                    self.current_wavelengths,
                    results["baseline"],
                    pen=pg.mkPen(
                        color=(150, 150, 150),
                        width=1.0,
                        style=Qt.PenStyle.DashLine,
                    ),
                )
            except (TypeError, ValueError):
                pass

        if (
            results.get("peak_wavelengths") is not None
            and getattr(results["peak_wavelengths"], "size", 0) > 0
        ):
            self.current_peaks_wl = results["peak_wavelengths"]
            self.current_peaks_int = results["peak_intensities"]
            # Share peak data with ResultsPanel for zoom plot - use setattr to bypass type checking
            setattr(self.results_panel, '_peaks', [self.current_peaks_wl, self.current_peaks_int])
            self.plot_widget.plot(
                self.current_peaks_wl,
                self.current_peaks_int,
                pen=None,
                symbol="o",
                symbolBrush="r",
                symbolPen="r",
                symbolSize=6,
            )

        if mode != "preprocess" and results.get("annotations"):
            for ann in results["annotations"]:
                text_color = QColor("red") if ann["is_top"] else QColor("black")
                line = pg.InfiniteLine(
                    pos=ann["x"],
                    angle=90,
                    movable=False,
                    pen=pg.mkPen(color=(200, 200, 200), style=Qt.PenStyle.DashLine),
                )
                self.plot_widget.addItem(line)
                text = pg.TextItem(ann["text"], color=text_color, anchor=(-0.1, 0.5))
                text.setAngle(90)
                text.setPos(ann["x"], ann["y"])
                self.plot_widget.addItem(text)

        # Zoom handling is delegated to ResultsPanel's ROI; avoid re-plotting here.

        table_data = (
            []
            if mode == "preprocess"
            else (
                results.get("validation_table")
                if results.get("validation_table")
                else results.get("prediction_table", [])
            )
        )

    @Slot(int)
    def toggle_zoom_mode(self, state: int):
        pass  # Removed functionality as zoom mode checkbox no longer exists

    def custom_mouse_double_click(self, event):
        self.reset_main_view()
        event.accept()

    # Removed on_slider_value_changed since we now use input field

    def trigger_interactive_analysis(self):
        idx = self.tabs.currentIndex()
        if idx == getattr(self, "predict_tab_index", -1):
            # Fast feedback while tuning controls: run preview preprocessing
            if self.raw_asc_content:
                self.previewRequested.emit(self.get_input_data())
    # Do not auto-run full analysis on validate tab; only on explicit button click

    @Slot(dict)
    def update_preview(self, results: dict[str, Any]):
        """Handle preview (preprocessing-only) results: plot spectrum and optional baseline.
        No peaks/annotations/tables expected here.
        """
        # Delegate UI updates to ResultsPanel; only sync local state used by overlays/crosshair
        self.last_results = results
        self.export_button.setEnabled(False)
        self._sync_state_from_results(results)
        # Table widget has been removed

    @Slot(str)
    def show_error(self, error_message: str):
        print(error_message)
        self.status_label.setText("Terjadi error! Lihat konsol.")

    def closeEvent(self, event):
        self.worker_thread.quit()
        self.worker_thread.wait()
        event.accept()

    def _on_full_finished_enable_export(self, results: dict[str, Any]):
        self.last_results = results
        has_table = bool(results.get("prediction_table") or results.get("validation_table"))
        self.export_button.setEnabled(has_table)

    @Slot(dict)
    def _sync_state_from_results(self, results: dict[str, Any]):
        try:
            wl = results.get("wavelengths")
            y = results.get("spectrum_data")
            self.current_wavelengths = None if wl is None else np.asarray(wl)
            self.current_intensities = None if y is None else np.asarray(y)
            pwl = results.get("peak_wavelengths")
            pint = results.get("peak_intensities")
            self.current_peaks_wl = None if pwl is None else np.asarray(pwl)
            self.current_peaks_int = None if pint is None else np.asarray(pint)
            bw = results.get("baseline_warning")
            if bw:
                self.status_label.setText(str(bw))
        except (TypeError, ValueError, AttributeError):
            # Keep previous state on any unexpected mismatch
            pass

    def ensure_zoom_region(self, xmin: float, xmax: float):
        if xmin >= xmax:
            return
        if self.region is None:
            width = xmax - xmin
            start = xmin + 0.25 * width
            end = xmin + 0.55 * width
            self.region = pg.LinearRegionItem(
                values=(start, end), orientation="vertical", movable=True
            )
            self.region.setBounds([xmin, xmax])
            self._region_proxy = pg.SignalProxy(
                self.region.sigRegionChanged,
                rateLimit=30,
                slot=lambda _: self.on_region_changed(),
            )
            self.plot_widget.addItem(self.region)
        else:
            self.region.setBounds([xmin, xmax])

    def on_region_changed(self):
        if (
            self.current_wavelengths is None
            or self.current_intensities is None
            or self.region is None
        ):
            return
        region_vals = self.region.getRegion()
        try:
            x0, x1 = map(float, region_vals)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        if x0 > x1:
            x0, x1 = x1, x0
        wl = np.asarray(self.current_wavelengths)
        intens = np.asarray(self.current_intensities)
        mask = (wl >= x0) & (wl <= x1)
        # Use the separate zoom_plot_widget to display zoomed data
        if not np.any(mask):
            self.status_label.setText(
                f"Zoom: Tidak ada data dalam {x0:.2f}-{x1:.2f} nm"
            )
            return
        # No need to adjust the main plot's view anymore, the zoom plot is updated via signals
        # Save the current peaks for reference in the zoom plot
        if self.current_peaks_wl is not None and self.current_peaks_int is not None:
            pwl = np.asarray(self.current_peaks_wl)
            pint = np.asarray(self.current_peaks_int)
            pmask = (pwl >= x0) & (pwl <= x1)
            # This is now handled by the update_zoom_plot method
        if getattr(self, "overlays", None):
            for ov in self.overlays:
                ow = ov.get("wl")
                oy = ov.get("y")
                pen = ov.get("pen", pg.mkPen("c", width=1.0))
                if ow is None or oy is None:
                    continue
                omask = (ow >= x0) & (ow <= x1)
                if np.any(omask):
                    pass  # Placeholder for future logic if needed

    def reset_main_view(self):
        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            vb = plot_item.getViewBox()
            if vb is not None:
                vb.autoRange()

    def _setup_crosshair(self, plot_widget: pg.PlotWidget, which: str):
        vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color=(180, 0, 0), width=1))
        hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color=(180, 0, 0), width=1))
        label = pg.TextItem(color=(10, 10, 10))
        plot_widget.addItem(vline)
        plot_widget.addItem(hline)
        plot_widget.addItem(label)
        scene = plot_widget.scene()
        if which == "main":
            self._main_vline, self._main_hline, self._main_label = vline, hline, label
            if hasattr(scene, "sigMouseMoved"):
                self._mouse_proxy_main = pg.SignalProxy(
                    getattr(scene, "sigMouseMoved"), rateLimit=60, slot=self._on_mouse_moved_main
                )
        else:
            self._zoom_vline, self._zoom_hline, self._zoom_label = vline, hline, label
            if hasattr(scene, "sigMouseMoved"):
                self._mouse_proxy_zoom = pg.SignalProxy(
                    getattr(scene, "sigMouseMoved"), rateLimit=60, slot=self._on_mouse_moved_zoom
                )

    def _on_mouse_moved_main(self, evt):
        if self.current_wavelengths is None or self.current_intensities is None:
            return
        if self._main_vline is None or self._main_hline is None or self._main_label is None:
            return
        pos = evt[0]
        vb = self.plot_widget.getViewBox()
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x = float(mouse_point.x())
            y = float(np.interp(x, self.current_wavelengths, self.current_intensities))
            self._main_vline.setPos(x)
            self._main_hline.setPos(y)
            self._main_label.setText(f"x={x:.2f}, y={y:.3g}")
            self._main_label.setPos(x, y)

    def _on_mouse_moved_zoom(self, evt):
        if self.current_wavelengths is None or self.current_intensities is None:
            return
        if self._zoom_vline is None or self._zoom_hline is None or self._zoom_label is None:
            return
        pos = evt[0]
        # Removed zoom_plot_widget logic as it no longer exists
        mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)
        x = float(mouse_point.x())
        y = float(np.interp(x, self.current_wavelengths, self.current_intensities))
        self._zoom_vline.setPos(x)
        self._zoom_hline.setPos(y)
        self._zoom_label.setText(f"x={x:.2f}, y={y:.3g}")
        self._zoom_label.setPos(x, y)

    def add_overlay_spectrum(self):
        if self.current_wavelengths is None or self.current_intensities is None:
            self.status_label.setText("Jalankan analisis dahulu sebelum overlay.")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Buka File ASC untuk Overlay", "", "ASC Files (*.asc);;All Files (*)"
        )
        if not filename:
            return
        try:
            with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            data = self.parse_asc_content(text)
        except (OSError, ValueError) as e:
            self.status_label.setText(f"Gagal memuat overlay: {e}")
            return
        wl = data[:, 0].astype(float)
        y = data[:, 1].astype(float)
        if np.max(y) > 0:
            y = y / np.max(y) * np.max(self.current_intensities)
        colors = ["c", "m", "y", "g", "r"]
        pen = pg.mkPen(colors[len(self.overlays) % len(colors)], width=1.0)
        self.plot_widget.plot(wl, y, pen=pen)
        self.overlays.append({"wl": wl, "y": y, "pen": pen})
        # Keep zoom in sync with ResultsPanel ROI; don't replot here
        if hasattr(self, "results_panel") and self.results_panel is not None:
            self.results_panel.on_region_changed()
            
    def _on_preview_requested_with_zoom(self, data):
        # Called after a preview request, update the zoom plot
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._update_zoom_with_current_region())
        timer.start(500)  # Short delay to ensure data is processed
    
    def _on_analyze_requested_with_zoom(self, data):
        # Called after analysis request, update the zoom plot
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._update_zoom_with_current_region())
        timer.start(500)  # Short delay to ensure data is processed
    
    def _update_zoom_with_current_region(self):
        """Update zoom plot with current region if available"""
        if self.region is not None and hasattr(self.results_panel, '_data_arrays') and self.results_panel._data_arrays is not None:
            region = self.region.getRegion()
            if region:
                x0, x1 = region
                print(f"Forcing zoom update with region: {x0}, {x1}")
                self.results_panel.update_zoom_plot(self.zoom_plot_widget, x0, x1)
