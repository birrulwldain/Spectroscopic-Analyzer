from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

# Force pyqtgraph to use PySide6 to avoid mixing Qt bindings (defensive)
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
import pyqtgraph as pg

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
        QTableWidgetItem,
        QHeaderView,
        QTabWidget,
        QFormLayout,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QSlider,
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
        self.setGeometry(100, 100, 1400, 900)

        # State
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
        self.zoom_plot_widget = results_panel.zoom_plot_widget
        self.radial_plot_widget = results_panel.radial_plot_widget
        self.table_widget = results_panel.table_widget
        self.zoom_mode_checkbox = results_panel.zoom_mode_checkbox
        self.results_panel = results_panel

        # Wire signals
        self.file_button.clicked.connect(self.open_file_dialog)
        # Bridge ControlPanel signals to MainWindow signals/handlers
        left_panel.previewRequested.connect(
            lambda d: self.previewRequested.emit({**d, "asc_content": self.raw_asc_content})
        )
        left_panel.analysisRequested.connect(self._on_analysis_requested)
        self.prominence_slider.valueChanged.connect(self.on_slider_value_changed)
        self.zoom_mode_checkbox.stateChanged.connect(self.toggle_zoom_mode)
        results_panel.btn_reset.clicked.connect(results_panel.on_reset_clicked)
        self.export_button.clicked.connect(self.export_results_to_xlsx)
        self.overlay_button.clicked.connect(self.add_overlay_spectrum)
        left_panel.batch_button.clicked.connect(
            lambda: self.open_batch_dialog(self._left_panel_ref.get_parameters())
        )

        # Root layout
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.addWidget(left_panel)
        root_layout.addWidget(results_panel)
        self.setCentralWidget(root)

        # Attach double-click reset on main view
        vb = self.plot_widget.getViewBox()
        if vb is not None:
            vb.mouseDoubleClickEvent = self.custom_mouse_double_click

        # Crosshairs
        self._setup_crosshair(self.plot_widget, which="main")
        self._setup_crosshair(self.zoom_plot_widget, which="zoom")

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
        # Show in results table
        self.table_widget.clear()
        if df is None or getattr(df, 'empty', False):
            self.table_widget.setRowCount(0)
            self.table_widget.setColumnCount(0)
            return
        headers = list(map(str, df.columns.tolist()))
        self.table_widget.setRowCount(int(len(df)))
        self.table_widget.setColumnCount(int(len(headers)))
        self.table_widget.setHorizontalHeaderLabels(headers)
        for i, (_, row) in enumerate(df.iterrows()):
            for j, key in enumerate(headers):
                self.table_widget.setItem(i, j, QTableWidgetItem(str(row.get(key, ""))))
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

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

    def open_file_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Buka File ASC", "", "ASC Files (*.asc);;All Files (*)"
        )
        if not filename:
            return
        encodings = ["utf-8", "utf-16", "latin-1"]
        last_err: Exception | None = None
        self.raw_asc_content = None
        for enc in encodings:
            try:
                with open(filename, "r", encoding=enc, errors="strict") as f:
                    self.raw_asc_content = f.read()
                break
            except (UnicodeDecodeError, OSError) as e:
                last_err = e
        if self.raw_asc_content is None:
            self.status_label.setText(f"Gagal membaca file: {last_err}")
            return

        base = os.path.basename(filename)
        self.status_label.setText(f"File: {base}")
        self.last_results = None
        self.export_button.setEnabled(False)

        try:
            data = self.parse_asc_content(self.raw_asc_content)
        except (ValueError, IndexError) as e:
            self.table_widget.clear()
            self.table_widget.setRowCount(0)
            self.plot_widget.clear()
            self.zoom_plot_widget.clear()
            self.status_label.setText(f"Format file tidak valid: {e}")
            return

        self.current_wavelengths = data[:, 0]
        self.current_intensities = data[:, 1]
        self.current_peaks_wl = None
        self.current_peaks_int = None
        # Trigger a quick preview so ResultsPanel draws the loaded signal
        self.previewRequested.emit(self.get_input_data())

    def get_input_data(self) -> dict[str, Any]:
        def to_float(s: str | None):
            return float(s) if s else None

        def to_int(s: str | None):
            return int(s) if s else None

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
        self.previewRequested.emit(data)

    def start_validation(self):
        if not self.raw_asc_content:
            return
        input_data = self.get_input_data()
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
        self.zoom_plot_widget.clear()
        self.radial_plot_widget.clear()
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
        if table_data:
            headers = list(table_data[0].keys())
            self.table_widget.setRowCount(len(table_data))
            self.table_widget.setColumnCount(len(headers))
            self.table_widget.setHorizontalHeaderLabels(headers)
            for i, row_data in enumerate(table_data):
                for j, key in enumerate(headers):
                    self.table_widget.setItem(i, j, QTableWidgetItem(str(row_data[key])))
            self.table_widget.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch
            )
        else:
            self.table_widget.clear()
            self.table_widget.setRowCount(0)
            self.table_widget.setColumnCount(0)

        rp = results.get("radial_profile")
        if rp is not None and len(rp) > 0:
            x_idx = np.arange(len(rp))
            self.radial_plot_widget.plot(x_idx, rp, pen=pg.mkPen("m", width=1.5))
            self.radial_plot_widget.setTitle("Profil Radial (Abel)")
        else:
            err = results.get("radial_profile_error")
            title = (
                "Profil Radial (tidak tersedia)" if not err else f"Profil Radial (error: {err})"
            )
            self.radial_plot_widget.setTitle(title)

    @Slot(int)
    def toggle_zoom_mode(self, state: int):
        is_on = state != 0
        # Delegate to ResultsPanel ROI-based zoom
        if hasattr(self, "results_panel") and self.results_panel is not None:
            self.results_panel.on_zoom_toggle(is_on)
            self.results_panel.zoom_plot_widget.setVisible(is_on)
            if is_on:
                self.results_panel.on_region_changed()

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
        # Clear table for preview (ResultsPanel already manages plots)
        self.table_widget.clear()
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(0)

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
        self.zoom_plot_widget.clear()
        if not np.any(mask):
            self.zoom_plot_widget.setTitle(
                f"Plot Zoom (Tidak ada data dalam {x0:.2f}-{x1:.2f} nm)"
            )
            return
        self.zoom_plot_widget.setTitle(f"Plot Zoom ({x0:.2f} - {x1:.2f} nm)")
        self.zoom_plot_widget.plot(wl[mask], intens[mask], pen=pg.mkPen("g", width=1.5))
        if self.current_peaks_wl is not None and self.current_peaks_int is not None:
            pwl = np.asarray(self.current_peaks_wl)
            pint = np.asarray(self.current_peaks_int)
            pmask = (pwl >= x0) & (pwl <= x1)
            if np.any(pmask):
                self.zoom_plot_widget.plot(
                    pwl[pmask],
                    pint[pmask],
                    pen=None,
                    symbol="o",
                    symbolBrush="r",
                    symbolPen="r",
                    symbolSize=6,
                )
        if getattr(self, "overlays", None):
            for ov in self.overlays:
                ow = ov.get("wl")
                oy = ov.get("y")
                pen = ov.get("pen", pg.mkPen("c", width=1.0))
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
        vb = self.zoom_plot_widget.getViewBox()
        if self.zoom_plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
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
