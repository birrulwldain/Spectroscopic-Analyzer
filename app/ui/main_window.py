from __future__ import annotations
import os
import numpy as np
import pandas as pd

# Force pyqtgraph to use PySide6 to avoid mixing Qt bindings (defensive)
os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PySide6')
import pyqtgraph as pg

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:  # for type checkers only
    from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt, QTimer
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
        QFileDialog, QLabel, QLineEdit, QGroupBox, QTableWidget, QTableWidgetItem,
        QHeaderView, QTabWidget, QFormLayout, QCheckBox, QComboBox, QDoubleSpinBox, QSlider
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

from app.ui.worker import Worker


class MainWindow(QMainWindow):
    analyzeRequested = Signal(dict)

    worker_thread: 'QThread'
    worker: 'Worker'
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
        self.setGeometry(100, 100, 1400, 900)
        self.raw_asc_content = None
        self.last_results = None
        self.current_wavelengths = None
        self.current_intensities = None
        self.current_peaks_wl = None
        self.current_peaks_int = None
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
        self.overlays = []

        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(400)
        self.debounce_timer.timeout.connect(self.trigger_interactive_analysis)

        self.worker_thread = QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self.worker_thread)
        self.analyzeRequested.connect(self.worker.run_analysis)
        self.worker.finished.connect(self.update_results)
        self.worker.error.connect(self.show_error)
        self.worker_thread.start()

        self.setup_ui()

    def setup_ui(self):
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)

        file_group = QGroupBox("1. Unggah Data")
        file_layout = QVBoxLayout(file_group)
        self.file_button = QPushButton("Pilih File .asc")
        self.file_button.clicked.connect(self.open_file_dialog)
        self.status_label = QLabel("Silakan pilih file.")
        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.status_label)

        self.tabs = QTabWidget()
        predict_tab = QWidget(); predict_layout = QVBoxLayout(predict_tab)
        self.predict_button = QPushButton("Prediksi Elemen")
        self.predict_button.clicked.connect(self.start_prediction)
        predict_layout.addWidget(self.predict_button)

        validate_tab = QWidget(); validate_layout = QVBoxLayout(validate_tab)
        self.gt_input = QLineEdit(); self.gt_input.setPlaceholderText("Contoh: Fe, Si, Mg")
        self.validate_button = QPushButton("Jalankan Validasi")
        self.validate_button.clicked.connect(self.start_validation)
        validate_layout.addWidget(QLabel("Ground Truth Elements:"))
        validate_layout.addWidget(self.gt_input)
        validate_layout.addWidget(self.validate_button)

        self.predict_tab_index = self.tabs.addTab(predict_tab, "Prediksi Cepat")
        self.validate_tab_index = self.tabs.addTab(validate_tab, "Validasi Model")

        param_group = QGroupBox("Parameter Opsional")
        param_layout = QFormLayout(param_group)
        self.prominence_slider = QSlider(Qt.Orientation.Horizontal)
        self.prominence_slider.setMinimum(1)
        self.prominence_slider.setMaximum(5000)
        self.prominence_slider.setValue(100)
        self.prominence_label = QLabel("")
        self.prominence_slider.valueChanged.connect(self.on_slider_value_changed)
        self.prominence_label.setText(f"{self.prominence_slider.value()/10000.0:.4f}")
        prom_row = QHBoxLayout()
        prom_row.addWidget(self.prominence_slider)
        prom_row.addWidget(self.prominence_label)
        self.distance_input = QLineEdit("8")
        self.height_input = QLineEdit(); self.height_input.setPlaceholderText("Opsional, contoh: 0.1")
        self.width_input = QLineEdit(); self.width_input.setPlaceholderText("Opsional, contoh: 20")
        self.threshold_input = QLineEdit("0.6")
        param_layout.addRow("Prominence:", prom_row)
        param_layout.addRow("Distance:", self.distance_input)
        param_layout.addRow("Height:", self.height_input)
        param_layout.addRow("Width:", self.width_input)
        param_layout.addRow("Prediction Threshold:", self.threshold_input)

        self.baseline_switch = QCheckBox("Aktifkan Koreksi Baseline")
        self.baseline_overlay_switch = QCheckBox("Tampilkan Overlay Baseline")
        self.raw_resolution_switch = QCheckBox("Gunakan resolusi asli (tanpa resampling)")
        self.raw_resolution_switch.setChecked(True)
        self.abel_switch = QCheckBox("Hitung Inversi Abel (basex)")
        self.lam_input = QLineEdit("100000")
        self.p_input = QLineEdit("0.01")
        self.niter_input = QLineEdit("10")
        param_layout.addRow(self.raw_resolution_switch)
        param_layout.addRow(self.baseline_switch)
        param_layout.addRow(self.baseline_overlay_switch)
        param_layout.addRow(self.abel_switch)
        param_layout.addRow("Lambda (lam):", self.lam_input)
        param_layout.addRow("Asymmetry (p):", self.p_input)
        param_layout.addRow("Iterations (niter):", self.niter_input)

        self.smooth_switch = QCheckBox("Smoothing (Savitzky–Golay)")
        self.sg_window_input = QLineEdit("11")
        self.sg_poly_input = QLineEdit("3")
        param_layout.addRow(self.smooth_switch)
        param_layout.addRow("SG Window (odd):", self.sg_window_input)
        param_layout.addRow("SG Polyorder:", self.sg_poly_input)

        self.norm_combo = QComboBox(); self.norm_combo.addItems(["None", "Max", "Area"])
        param_layout.addRow("Normalization:", self.norm_combo)

        self.shift_spin = QDoubleSpinBox(); self.shift_spin.setRange(-10.0, 10.0)
        self.shift_spin.setDecimals(3); self.shift_spin.setSingleStep(0.01); self.shift_spin.setValue(0.0)
        param_layout.addRow("Wavelength shift (nm):", self.shift_spin)

        self.export_button = QPushButton("Ekspor Puncak Berlabel ke XLSX")
        self.export_button.clicked.connect(self.export_results_to_xlsx)
        self.export_button.setEnabled(False)
        self.overlay_button = QPushButton("Tambah Overlay Spektrum")
        self.overlay_button.clicked.connect(self.add_overlay_spectrum)

        left_layout.addWidget(file_group)
        left_layout.addWidget(self.tabs)
        left_layout.addWidget(param_group)
        left_layout.addWidget(self.overlay_button)
        left_layout.addWidget(self.export_button)
        left_layout.addStretch()
        left_container.setMaximumWidth(450)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)

        plot_group = QGroupBox("Hasil Interpretasi Grafik")
        plot_group_layout = QVBoxLayout(plot_group)
        self.plot_widget = pg.PlotWidget(); self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.getViewBox().mouseDoubleClickEvent = self.custom_mouse_double_click
        self.plot_widget.setTitle("Plot Utama (Overview)")
        plot_group_layout.addWidget(self.plot_widget)

        self.zoom_plot_widget = pg.PlotWidget(); self.zoom_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.zoom_plot_widget.setTitle("Plot Zoom (Area Terpilih)")
        plot_group_layout.addWidget(self.zoom_plot_widget)

        self.radial_plot_widget = pg.PlotWidget(); self.radial_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.radial_plot_widget.setTitle("Profil Radial (Abel)")
        plot_group_layout.addWidget(self.radial_plot_widget)

        plot_controls_layout = QHBoxLayout()
        self.zoom_mode_checkbox = QCheckBox("Tampilkan Seleksi Zoom")
        self.zoom_mode_checkbox.setChecked(True)
        self.zoom_mode_checkbox.stateChanged.connect(self.toggle_zoom_mode)
        btn_reset = QPushButton("Reset View"); btn_reset.clicked.connect(self.reset_main_view)
        plot_controls_layout.addWidget(self.zoom_mode_checkbox)
        plot_controls_layout.addStretch()
        plot_controls_layout.addWidget(btn_reset)
        plot_group_layout.addLayout(plot_controls_layout)

        table_group = QGroupBox("Tabel Hasil")
        table_layout = QVBoxLayout(table_group)
        self.table_widget = QTableWidget()
        table_layout.addWidget(self.table_widget)

        right_layout.addWidget(plot_group, 2)
        right_layout.addWidget(table_group, 1)

        root = QWidget(); root_layout = QHBoxLayout(root)
        root_layout.addWidget(left_container)
        root_layout.addWidget(right_container)
        self.setCentralWidget(root)

        self._setup_crosshair(self.plot_widget, which='main')
        self._setup_crosshair(self.zoom_plot_widget, which='zoom')

    def parse_asc_content(self, text: str) -> np.ndarray:
        rows = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith(("#", "//", "%", ";")):
                continue
            line = line.replace(",", " ")
            parts = [p for p in line.split() if p]
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0]); y = float(parts[1])
                rows.append((x, y))
            except ValueError:
                continue
        if not rows:
            raise ValueError("Tidak menemukan pasangan data numerik (x y).")
        return np.asarray(rows, dtype=float)

    def open_file_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Buka File ASC", "", "ASC Files (*.asc);;All Files (*)")
        if filename:
            encodings = ["utf-8", "utf-16", "latin-1"]
            last_err: Exception | None = None
            self.raw_asc_content = None
            for enc in encodings:
                try:
                    with open(filename, 'r', encoding=enc, errors="strict") as f:
                        self.raw_asc_content = f.read()
                    break
                except (UnicodeDecodeError, OSError) as e:
                    last_err = e
            if self.raw_asc_content is None:
                self.status_label.setText(f"Gagal membaca file: {last_err}")
                return

            base = os.path.basename(filename)
            self.status_label.setText(f"File: {base}")
            self.last_results = None; self.export_button.setEnabled(False)

            try:
                data = self.parse_asc_content(self.raw_asc_content)
            except (ValueError, IndexError) as e:
                self.table_widget.clear(); self.table_widget.setRowCount(0)
                self.plot_widget.clear(); self.zoom_plot_widget.clear()
                self.status_label.setText(f"Format file tidak valid: {e}")
                return
            self.current_wavelengths = data[:, 0]; self.current_intensities = data[:, 1]
            self.current_peaks_wl = None; self.current_peaks_int = None
            self.plot_widget.clear(); self.zoom_plot_widget.clear(); self.table_widget.clear(); self.table_widget.setRowCount(0)
            self.plot_widget.plot(self.current_wavelengths, self.current_intensities, pen=pg.mkPen('k', width=0.5))
            self.plot_widget.setTitle(f"Pratinjau Sinyal: {base}")
            self.ensure_zoom_region(float(np.min(self.current_wavelengths)), float(np.max(self.current_wavelengths)))
            self.on_region_changed()

    def get_input_data(self):
        def to_float(s): return float(s) if s else None
        def to_int(s): return int(s) if s else None
        prominence_val = float(self.prominence_slider.value()) / 10000.0
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
        if self.raw_asc_content:
            self.analyzeRequested.emit(self.get_input_data())

    def start_validation(self):
        if self.raw_asc_content:
            input_data = self.get_input_data()
            input_data["ground_truth_elements"] = [el.strip() for el in self.gt_input.text().split(',') if el.strip()]
            self.analyzeRequested.emit(input_data)

    def export_results_to_xlsx(self):
        if not self.last_results:
            self.status_label.setText("Tidak ada data untuk diekspor.")
            return
        table_data = self.last_results.get("validation_table") if self.last_results.get("validation_table") else self.last_results.get("prediction_table", [])
        if not table_data:
            self.status_label.setText("Tidak ada data tabel untuk diekspor.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Simpan Hasil Excel", "", "Excel Files (*.xlsx)")
        if filename:
            try:
                pd.DataFrame(table_data).to_excel(filename, index=False, sheet_name="Hasil Analisis")
                self.status_label.setText(f"Hasil berhasil disimpan di {filename}")
            except (OSError, ValueError) as e:
                self.status_label.setText(f"Gagal menyimpan file: {e}")

    @Slot(dict)
    def update_results(self, results):
        self.last_results = results
        self.export_button.setEnabled(bool(results.get("prediction_table") or results.get("validation_table")))

        self.current_wavelengths = results["wavelengths"]
        self.current_intensities = results["spectrum_data"]
        self.current_peaks_wl = None
        self.current_peaks_int = None

        self.plot_widget.clear()
        self.zoom_plot_widget.clear()
        self.radial_plot_widget.clear()
        self.overlays.clear()

        self.plot_widget.setTitle("Hasil Analisis")
        self.plot_widget.plot(self.current_wavelengths, self.current_intensities, pen=pg.mkPen('b', width=1.5))
        if self.baseline_overlay_switch.isChecked() and results.get("baseline") is not None:
            try:
                self.plot_widget.plot(
                    self.current_wavelengths,
                    results["baseline"],
                    pen=pg.mkPen(color=(150, 150, 150), width=1.0, style=Qt.PenStyle.DashLine),
                )
            except (TypeError, ValueError):
                pass

        if results.get("peak_wavelengths") is not None and getattr(results["peak_wavelengths"], 'size', 0) > 0:
            self.current_peaks_wl = results["peak_wavelengths"]
            self.current_peaks_int = results["peak_intensities"]
            self.plot_widget.plot(
                self.current_peaks_wl,
                self.current_peaks_int,
                pen=None,
                symbol='o',
                symbolBrush='r',
                symbolPen='r',
                symbolSize=6,
            )

        if results.get("annotations"):
            for ann in results["annotations"]:
                text_color = QColor('red') if ann["is_top"] else QColor('black')
                line = pg.InfiniteLine(
                    pos=ann["x"], angle=90, movable=False,
                    pen=pg.mkPen(color=(200, 200, 200), style=Qt.PenStyle.DashLine)
                )
                self.plot_widget.addItem(line)
                text = pg.TextItem(ann["text"], color=text_color, anchor=(-0.1, 0.5))
                text.setAngle(90)
                text.setPos(ann["x"], ann["y"])
                self.plot_widget.addItem(text)

        if self.current_wavelengths is not None and len(self.current_wavelengths) > 1:
            self.ensure_zoom_region(float(self.current_wavelengths[0]), float(self.current_wavelengths[-1]))
            self.on_region_changed()

        table_data = results.get("validation_table") if results.get("validation_table") else results.get("prediction_table", [])
        if table_data:
            headers = list(table_data[0].keys())
            self.table_widget.setRowCount(len(table_data))
            self.table_widget.setColumnCount(len(headers))
            self.table_widget.setHorizontalHeaderLabels(headers)
            for i, row_data in enumerate(table_data):
                for j, key in enumerate(headers):
                    self.table_widget.setItem(i, j, QTableWidgetItem(str(row_data[key])))
            self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        else:
            self.table_widget.clear()
            self.table_widget.setRowCount(0)
            self.table_widget.setColumnCount(0)

        rp = results.get("radial_profile")
        if rp is not None and len(rp) > 0:
            x_idx = np.arange(len(rp))
            self.radial_plot_widget.plot(x_idx, rp, pen=pg.mkPen('m', width=1.5))
            self.radial_plot_widget.setTitle("Profil Radial (Abel)")
        else:
            err = results.get("radial_profile_error")
            title = "Profil Radial (tidak tersedia)" if not err else f"Profil Radial (error: {err})"
            self.radial_plot_widget.setTitle(title)

    @Slot(int)
    def toggle_zoom_mode(self, state):
        is_on = (state != 0)
        if self.region is not None:
            self.region.setVisible(is_on)
        self.zoom_plot_widget.setVisible(is_on)
        if is_on:
            self.on_region_changed()

    def custom_mouse_double_click(self, event):
        self.reset_main_view()
        event.accept()

    def on_slider_value_changed(self, value: int):
        pval = float(value) / 10000.0
        self.prominence_label.setText(f"{pval:.4f}")
        if self.debounce_timer.isActive():
            self.debounce_timer.stop()
        self.debounce_timer.start()

    def trigger_interactive_analysis(self):
        idx = self.tabs.currentIndex()
        if idx == getattr(self, 'predict_tab_index', -1):
            self.start_prediction()
        elif idx == getattr(self, 'validate_tab_index', -1):
            self.start_validation()

    @Slot(str)
    def show_error(self, error_message):
        print(error_message)
        self.status_label.setText("Terjadi error! Lihat konsol.")

    def closeEvent(self, event):
        self.worker_thread.quit()
        self.worker_thread.wait()
        event.accept()

    def ensure_zoom_region(self, xmin: float, xmax: float):
        if xmin >= xmax:
            return
        if self.region is None:
            width = (xmax - xmin)
            start = xmin + 0.25 * width
            end = xmin + 0.55 * width
            self.region = pg.LinearRegionItem(values=(start, end), orientation='vertical', movable=True)
            self.region.setBounds([xmin, xmax])
            self._region_proxy = pg.SignalProxy(self.region.sigRegionChanged, rateLimit=30, slot=lambda _: self.on_region_changed())
            self.plot_widget.addItem(self.region)
        else:
            self.region.setBounds([xmin, xmax])

    def on_region_changed(self):
        if self.current_wavelengths is None or self.current_intensities is None or self.region is None:
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
        self.zoom_plot_widget.clear()
        if not np.any(mask):
            self.zoom_plot_widget.setTitle(f"Plot Zoom (Tidak ada data dalam {x0:.2f}-{x1:.2f} nm)")
            return
        self.zoom_plot_widget.setTitle(f"Plot Zoom ({x0:.2f} - {x1:.2f} nm)")
        self.zoom_plot_widget.plot(wl[mask], intens[mask], pen=pg.mkPen('g', width=1.5))
        if self.current_peaks_wl is not None and self.current_peaks_int is not None:
            pwl = np.asarray(self.current_peaks_wl); pint = np.asarray(self.current_peaks_int)
            pmask = (pwl >= x0) & (pwl <= x1)
            if np.any(pmask):
                self.zoom_plot_widget.plot(pwl[pmask], pint[pmask], pen=None, symbol='o', symbolBrush='r', symbolPen='r', symbolSize=6)
        if getattr(self, 'overlays', None):
            for ov in self.overlays:
                ow = ov.get("wl"); oy = ov.get("y"); pen = ov.get("pen", pg.mkPen('c', width=1.0))
                if ow is None or oy is None:
                    continue
                omask = (ow >= x0) & (ow <= x1)
                if np.any(omask):
                    self.zoom_plot_widget.plot(ow[omask], oy[omask], pen=pen)

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
        if which == 'main':
            self._main_vline, self._main_hline, self._main_label = vline, hline, label
            if hasattr(scene, 'sigMouseMoved'):
                self._mouse_proxy_main = pg.SignalProxy(getattr(scene, 'sigMouseMoved'), rateLimit=60, slot=self._on_mouse_moved_main)
        else:
            self._zoom_vline, self._zoom_hline, self._zoom_label = vline, hline, label
            if hasattr(scene, 'sigMouseMoved'):
                self._mouse_proxy_zoom = pg.SignalProxy(getattr(scene, 'sigMouseMoved'), rateLimit=60, slot=self._on_mouse_moved_zoom)

    def _on_mouse_moved_main(self, evt):
        if self.current_wavelengths is None or self.current_intensities is None:
            return
        if self._main_vline is None or self._main_hline is None or self._main_label is None:
            return
        pos = evt[0]
        vb = self.plot_widget.getViewBox()
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x = float(mouse_point.x()); y = float(np.interp(x, self.current_wavelengths, self.current_intensities))
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
        vb = self.zoom_plot_widget.getViewBox()
        if self.zoom_plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x = float(mouse_point.x()); y = float(np.interp(x, self.current_wavelengths, self.current_intensities))
            self._zoom_vline.setPos(x)
            self._zoom_hline.setPos(y)
            self._zoom_label.setText(f"x={x:.2f}, y={y:.3g}")
            self._zoom_label.setPos(x, y)

    def add_overlay_spectrum(self):
        if self.current_wavelengths is None or self.current_intensities is None:
            self.status_label.setText("Jalankan analisis dahulu sebelum overlay.")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Buka File ASC untuk Overlay",
            "",
            "ASC Files (*.asc);;All Files (*)",
        )
        if not filename:
            return
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            data = self.parse_asc_content(text)
        except (OSError, ValueError) as e:
            self.status_label.setText(f"Gagal memuat overlay: {e}")
            return
        wl = data[:, 0].astype(float)
        y = data[:, 1].astype(float)
        if np.max(y) > 0:
            y = y / np.max(y) * np.max(self.current_intensities)
        colors = ['c', 'm', 'y', 'g', 'r']
        pen = pg.mkPen(colors[len(self.overlays) % len(colors)], width=1.0)
        self.plot_widget.plot(wl, y, pen=pen)
        self.overlays.append({"wl": wl, "y": y, "pen": pen})
        self.on_region_changed()
