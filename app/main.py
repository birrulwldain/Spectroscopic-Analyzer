import sys
import os
import traceback
import pandas as pd
import numpy as np
import torch
from scipy.signal import find_peaks

from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QLineEdit, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QFormLayout, QCheckBox
)

# Support running as `python app/main.py` or `python -m app.main`
try:
    from app.model import load_assets, als_baseline_correction
    from app.processing import prepare_asc_data
except ModuleNotFoundError:
    # Add project root to sys.path then retry absolute imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from app.model import load_assets, als_baseline_correction
    from app.processing import prepare_asc_data

import pyqtgraph as pg
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


print("Memuat model dan aset-aset penting...")
model, element_map, target_wavelengths = load_assets()
class_names = list(element_map.keys())
print("Aset berhasil dimuat.")


def run_full_analysis(input_data: dict):
    spectrum_data = prepare_asc_data(input_data["asc_content"], target_wavelengths)
    if input_data.get("apply_baseline_correction"):
        lam = input_data.get("lam", 100000)
        p = input_data.get("p", 0.01)
        niter = input_data.get("niter", 10)
        spectrum_data = spectrum_data - als_baseline_correction(spectrum_data, lam, p, niter)

    peak_indices, _ = find_peaks(
        spectrum_data,
        prominence=input_data.get("prominence"),
        distance=input_data.get("distance"),
        height=input_data.get("height"),
        width=input_data.get("width")
    )

    input_tensor = torch.from_numpy(spectrum_data[np.newaxis, :, np.newaxis]).float()
    with torch.no_grad():
        output_logits = model(input_tensor)
        full_probabilities = torch.sigmoid(output_logits).squeeze(0).cpu().numpy()

    all_peaks_list = []
    threshold = input_data.get("threshold", 0.6)
    for peak_idx in peak_indices:
        prediction_at_peak = (full_probabilities[peak_idx] > threshold).astype(int)
        detected_indices = np.where(prediction_at_peak == 1)[0]
        if detected_indices.size > 0:
            elements_on_peak = [class_names[j] for j in detected_indices if class_names[j] != 'background']
            if elements_on_peak:
                all_peaks_list.append({
                    "wavelength": float(np.round(target_wavelengths[peak_idx], 2)),
                    "intensity": float(spectrum_data[peak_idx]),
                    "elements": elements_on_peak
                })

    sorted_peaks = sorted(all_peaks_list, key=lambda p: p['intensity'], reverse=True)

    annotations = []
    element_rank_counter = {}
    predicted_elements_with_locations = {}
    for peak in sorted_peaks:
        label_parts = []
        is_top_peak_overall = False
        for element in sorted(peak["elements"]):
            predicted_elements_with_locations.setdefault(element, []).append(peak["wavelength"])
            current_rank = element_rank_counter.get(element, 0) + 1
            element_rank_counter[element] = current_rank
            if current_rank <= 10:
                label_parts.append(f"{element} ({current_rank})")
                is_top_peak_overall = True
            else:
                label_parts.append(element)
        if label_parts:
            base_text = " ".join(label_parts)
            final_text = f"{base_text} {peak['wavelength']:.2f} nm"
            annotations.append({
                "x": peak["wavelength"], "y": peak["intensity"], "text": final_text,
                "is_top": is_top_peak_overall
            })

    prediction_table = []
    validation_table = []
    summary_metrics = {}
    predicted_elements_set = set(predicted_elements_with_locations.keys())
    for el in sorted(predicted_elements_set):
        locs = predicted_elements_with_locations.get(el, [])
        prediction_table.append({
            "Elemen": el,
            "Jumlah Puncak": len(locs),
            "Lokasi Puncak (nm)": '; '.join(map(str, sorted(locs)))
        })

    if input_data.get("ground_truth_elements"):
        ground_truth_set = set(input_data["ground_truth_elements"])
        tp_set = predicted_elements_set & ground_truth_set
        fp_set = predicted_elements_set - ground_truth_set
        fn_set = ground_truth_set - predicted_elements_set
        tp = len(tp_set); fp = len(fp_set); fn = len(fn_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        summary_metrics = {"Precision": f"{precision:.2%}", "Recall": f"{recall:.2%}", "F1-Score": f"{f1_score:.2%}", "TP": tp, "FP": fp, "FN": fn}
        for el in sorted(tp_set):
            validation_table.append({"Elemen": el, "Status": "True Positive", "Jumlah Puncak": len(predicted_elements_with_locations.get(el, [])), "Lokasi (nm)": '; '.join(map(str, sorted(predicted_elements_with_locations.get(el, []))))})
        for el in sorted(fp_set):
            validation_table.append({"Elemen": el, "Status": "False Positive", "Jumlah Puncak": len(predicted_elements_with_locations.get(el, [])), "Lokasi (nm)": '; '.join(map(str, sorted(predicted_elements_with_locations.get(el, []))))})
        for el in sorted(fn_set):
            validation_table.append({"Elemen": el, "Status": "False Negative", "Jumlah Puncak": 0, "Lokasi (nm)": "-"})

    return {
        "spectrum_data": spectrum_data,
        "wavelengths": target_wavelengths,
        "peak_wavelengths": np.array([p['wavelength'] for p in sorted_peaks], dtype=np.float32),
        "peak_intensities": np.array([p['intensity'] for p in sorted_peaks], dtype=np.float32),
        "annotations": annotations,
        "prediction_table": prediction_table,
        "validation_table": validation_table,
        "summary_metrics": summary_metrics,
    }


class Worker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    @Slot(dict)
    def run_analysis(self, input_data):
        try:
            results = run_full_analysis(input_data)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(f"Error: {e}\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    analyzeRequested = Signal(dict)

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

        self.worker_thread = QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self.worker_thread)
        self.analyzeRequested.connect(self.worker.run_analysis)
        self.worker.finished.connect(self.update_results)
        self.worker.error.connect(self.show_error)
        self.worker_thread.start()

        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout()

        left_container = QWidget(); left_layout = QVBoxLayout(); left_container.setLayout(left_layout)
        file_group = QGroupBox("1. Unggah Data"); file_layout = QVBoxLayout(); file_group.setLayout(file_layout)
        self.file_button = QPushButton("Pilih File .asc"); self.file_button.clicked.connect(self.open_file_dialog)
        self.status_label = QLabel("Silakan pilih file.")
        file_layout.addWidget(self.file_button); file_layout.addWidget(self.status_label)

        self.tabs = QTabWidget()
        predict_tab, validate_tab = QWidget(), QWidget()
        predict_layout = QVBoxLayout(); predict_tab.setLayout(predict_layout)
        self.predict_button = QPushButton("Prediksi Elemen"); self.predict_button.clicked.connect(self.start_prediction)
        predict_layout.addWidget(self.predict_button)

        validate_layout = QVBoxLayout(); validate_tab.setLayout(validate_layout)
        self.gt_input = QLineEdit(); self.gt_input.setPlaceholderText("Contoh: Fe, Si, Mg")
        self.validate_button = QPushButton("Jalankan Validasi"); self.validate_button.clicked.connect(self.start_validation)
        validate_layout.addWidget(QLabel("Ground Truth Elements:")); validate_layout.addWidget(self.gt_input); validate_layout.addWidget(self.validate_button)

        self.tabs.addTab(predict_tab, "Prediksi Cepat"); self.tabs.addTab(validate_tab, "Validasi Model")

        param_group = QGroupBox("Parameter Opsional"); param_layout = QFormLayout(); param_group.setLayout(param_layout)
        self.prominence_input = QLineEdit("0.01"); self.distance_input = QLineEdit("8")
        self.height_input = QLineEdit(); self.height_input.setPlaceholderText("Opsional, contoh: 0.1")
        self.width_input = QLineEdit(); self.width_input.setPlaceholderText("Opsional, contoh: 20")
        self.threshold_input = QLineEdit("0.6")
        param_layout.addRow("Prominence:", self.prominence_input)
        param_layout.addRow("Distance:", self.distance_input)
        param_layout.addRow("Height:", self.height_input)
        param_layout.addRow("Width:", self.width_input)
        param_layout.addRow("Prediction Threshold:", self.threshold_input)
        self.baseline_switch = QCheckBox("Aktifkan Koreksi Baseline")
        self.lam_input = QLineEdit("100000"); self.p_input = QLineEdit("0.01"); self.niter_input = QLineEdit("10")
        param_layout.addRow(self.baseline_switch)
        param_layout.addRow("Lambda (lam):", self.lam_input)
        param_layout.addRow("Asymmetry (p):", self.p_input)
        param_layout.addRow("Iterations (niter):", self.niter_input)

        self.export_button = QPushButton("Ekspor Puncak Berlabel ke XLSX")
        self.export_button.clicked.connect(self.export_results_to_xlsx)
        self.export_button.setEnabled(False)

        left_layout.addWidget(file_group)
        left_layout.addWidget(self.tabs)
        left_layout.addWidget(param_group)
        left_layout.addWidget(self.export_button)
        left_layout.addStretch()
        left_container.setMaximumWidth(450)

        right_container = QWidget(); right_layout = QVBoxLayout(); right_container.setLayout(right_layout)
        plot_group = QGroupBox("Hasil Interpretasi Grafik"); plot_group_layout = QVBoxLayout(); plot_group.setLayout(plot_group_layout)
        self.plot_widget = pg.PlotWidget(); self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.getViewBox().mouseDoubleClickEvent = self.custom_mouse_double_click
        self.plot_widget.setTitle("Plot Utama (Overview)"); plot_group_layout.addWidget(self.plot_widget)
        self.zoom_plot_widget = pg.PlotWidget(); self.zoom_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.zoom_plot_widget.setTitle("Plot Zoom (Area Terpilih)"); plot_group_layout.addWidget(self.zoom_plot_widget)
        plot_controls_layout = QHBoxLayout()
        self.zoom_mode_checkbox = QCheckBox("Tampilkan Seleksi Zoom"); self.zoom_mode_checkbox.setChecked(True)
        self.zoom_mode_checkbox.stateChanged.connect(self.toggle_zoom_mode)
        btn_reset = QPushButton("Reset View"); btn_reset.clicked.connect(self.reset_main_view)
        plot_controls_layout.addWidget(self.zoom_mode_checkbox); plot_controls_layout.addStretch(); plot_controls_layout.addWidget(btn_reset)
        plot_group_layout.addLayout(plot_controls_layout)

        table_group = QGroupBox("Tabel Hasil"); table_layout = QVBoxLayout(); table_group.setLayout(table_layout)
        self.table_widget = QTableWidget(); table_layout.addWidget(self.table_widget)

        right_layout.addWidget(plot_group, 2)
        right_layout.addWidget(table_group, 1)

        root = QWidget(); root_layout = QHBoxLayout(); root.setLayout(root_layout)
        root_layout.addWidget(left_container)
        root_layout.addWidget(right_container)
        self.setCentralWidget(root)

    def open_file_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Buka File ASC", "", "ASC Files (*.asc)")
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                self.raw_asc_content = f.read()
            self.status_label.setText(f"File: {filename.split('/')[-1]}")
            self.last_results = None; self.export_button.setEnabled(False)
            data = np.array([list(map(float, line.split())) for line in self.raw_asc_content.strip().split('\n') if len(line.split()) == 2])
            self.current_wavelengths = data[:, 0]; self.current_intensities = data[:, 1]
            self.current_peaks_wl = None; self.current_peaks_int = None
            self.plot_widget.clear(); self.zoom_plot_widget.clear(); self.table_widget.clear(); self.table_widget.setRowCount(0)
            self.plot_widget.plot(self.current_wavelengths, self.current_intensities, pen=pg.mkPen('k', width=0.5))
            self.plot_widget.setTitle(f"Pratinjau Sinyal: {filename.split('/')[-1]}")
            self.ensure_zoom_region(float(np.min(self.current_wavelengths)), float(np.max(self.current_wavelengths)))
            self.on_region_changed()

    def get_input_data(self):
        def to_float(s): return float(s) if s else None
        def to_int(s): return int(s) if s else None
        return {
            "asc_content": self.raw_asc_content,
            "prominence": to_float(self.prominence_input.text()),
            "distance": to_float(self.distance_input.text()),
            "height": to_float(self.height_input.text()),
            "width": to_float(self.width_input.text()),
            "threshold": to_float(self.threshold_input.text()),
            "apply_baseline_correction": self.baseline_switch.isChecked(),
            "lam": to_float(self.lam_input.text()),
            "p": to_float(self.p_input.text()),
            "niter": to_int(self.niter_input.text()),
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
            except Exception as e:
                self.status_label.setText(f"Gagal menyimpan file: {e}")

    @Slot(dict)
    def update_results(self, results):
        self.last_results = results
        self.export_button.setEnabled(bool(results.get("prediction_table") or results.get("validation_table")))

        self.current_wavelengths = results["wavelengths"]
        self.current_intensities = results["spectrum_data"]
        self.current_peaks_wl = None; self.current_peaks_int = None

        self.plot_widget.clear(); self.zoom_plot_widget.clear()
        self.plot_widget.setTitle("Hasil Analisis")
        self.plot_widget.plot(self.current_wavelengths, self.current_intensities, pen=pg.mkPen('b', width=1.5))
        if results.get("peak_wavelengths") is not None and getattr(results["peak_wavelengths"], 'size', 0) > 0:
            self.current_peaks_wl = results["peak_wavelengths"]
            self.current_peaks_int = results["peak_intensities"]
            self.plot_widget.plot(self.current_peaks_wl, self.current_peaks_int, pen=None, symbol='o', symbolBrush='r', symbolPen='r', symbolSize=6)
        if results.get("annotations"):
            for ann in results["annotations"]:
                text_color = QColor('red') if ann["is_top"] else QColor('black')
                line = pg.InfiniteLine(pos=ann["x"], angle=90, movable=False, pen=pg.mkPen(color=(200, 200, 200), style=Qt.PenStyle.DashLine))
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
            self.table_widget.setRowCount(len(table_data)); self.table_widget.setColumnCount(len(headers))
            self.table_widget.setHorizontalHeaderLabels(headers)
            for i, row_data in enumerate(table_data):
                for j, key in enumerate(headers):
                    self.table_widget.setItem(i, j, QTableWidgetItem(str(row_data[key])))
            self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        else:
            self.table_widget.clear(); self.table_widget.setRowCount(0); self.table_widget.setColumnCount(0)

    @Slot(int)
    def toggle_zoom_mode(self, state):
        """Show/hide zoom region and plot."""
        is_on = state == Qt.CheckState.Checked.value
        if self.region is not None:
            self.region.setVisible(is_on)
        self.zoom_plot_widget.setVisible(is_on)
        if is_on:
            self.on_region_changed()

    def custom_mouse_double_click(self, event):
        """Double-click handler to reset the main view box."""
        self.reset_main_view()
        event.accept()

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
            self.region.sigRegionChanged.connect(self.on_region_changed)
            self.plot_widget.addItem(self.region)
        else:
            self.region.setBounds([xmin, xmax])

    def on_region_changed(self):
        if self.current_wavelengths is None or self.current_intensities is None or self.region is None:
            return
        region_vals = self.region.getRegion()
        try:
            x0, x1 = map(float, region_vals)  # type: ignore[arg-type]
        except Exception:
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

    def reset_main_view(self):
        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            vb = plot_item.getViewBox()
            if vb is not None:
                vb.autoRange()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())