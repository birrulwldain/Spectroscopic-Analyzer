from __future__ import annotations
from typing import Any, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from PySide6.QtCore import Signal, Qt
    from PySide6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGroupBox,
        QLabel,
        QPushButton,
        QFormLayout,
        QLineEdit,
        QFileDialog,
        QTabWidget,
        QDoubleSpinBox,
        QComboBox,
        QCheckBox,
        QSlider,
    )
else:
    from PySide6 import QtCore as _QtCore  # type: ignore
    from PySide6 import QtWidgets as _QtWidgets  # type: ignore

    Signal = _QtCore.Signal
    Qt = _QtCore.Qt

    QWidget = _QtWidgets.QWidget
    QVBoxLayout = _QtWidgets.QVBoxLayout
    QHBoxLayout = _QtWidgets.QHBoxLayout
    QGroupBox = _QtWidgets.QGroupBox
    QLabel = _QtWidgets.QLabel
    QPushButton = _QtWidgets.QPushButton
    QFormLayout = _QtWidgets.QFormLayout
    QLineEdit = _QtWidgets.QLineEdit
    QFileDialog = _QtWidgets.QFileDialog
    QTabWidget = _QtWidgets.QTabWidget
    QDoubleSpinBox = _QtWidgets.QDoubleSpinBox
    QComboBox = _QtWidgets.QComboBox
    QCheckBox = _QtWidgets.QCheckBox
    QSlider = _QtWidgets.QSlider


class ControlPanel(QWidget):
    previewRequested = Signal(dict)
    analysisRequested = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # 1) File group + prominent Preview button
        file_group = QGroupBox("1. Unggah Data")
        file_layout = QVBoxLayout(file_group)
        self.file_button = QPushButton("Pilih File .asc")
        self.status_label = QLabel("Silakan pilih file.")
        self.preview_button = QPushButton("Terapkan & Pratinjau Pra-pemrosesan")
        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.status_label)
        file_layout.addWidget(self.preview_button)

        # 2) Pra-pemrosesan Sinyal
        pp_group = QGroupBox("2. Pra-pemrosesan Sinyal")
        pp_layout = QVBoxLayout(pp_group)
        pp_form = QFormLayout()

        self.shift_spin = QDoubleSpinBox()
        self.shift_spin.setRange(-10.0, 10.0)
        self.shift_spin.setDecimals(3)
        self.shift_spin.setSingleStep(0.01)
        self.shift_spin.setValue(0.0)

        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["None", "Max", "Area"])

        self.smooth_switch = QCheckBox("Smoothing (Savitzky–Golay)")
        self.sg_window_input = QLineEdit("11")
        self.sg_poly_input = QLineEdit("3")

        self.baseline_switch = QCheckBox("Aktifkan Koreksi Baseline")
        self.baseline_overlay_switch = QCheckBox("Tampilkan Overlay Baseline")
        self.raw_resolution_switch = QCheckBox("Gunakan resolusi asli (tanpa resampling)")
        self.raw_resolution_switch.setChecked(True)

        self.abel_switch = QCheckBox("Hitung Inversi Abel (basex)")
        self.lam_input = QLineEdit("100000")
        self.p_input = QLineEdit("0.01")
        self.niter_input = QLineEdit("10")

        pp_form.addRow("Wavelength shift (nm):", self.shift_spin)
        pp_form.addRow("Normalization:", self.norm_combo)
        pp_form.addRow(self.smooth_switch)
        pp_form.addRow("SG Window (odd):", self.sg_window_input)
        pp_form.addRow("SG Polyorder:", self.sg_poly_input)
        pp_form.addRow(self.raw_resolution_switch)
        pp_form.addRow(self.baseline_switch)
        pp_form.addRow(self.baseline_overlay_switch)
        pp_form.addRow(self.abel_switch)
        pp_form.addRow("Lambda (lam):", self.lam_input)
        pp_form.addRow("Asymmetry (p):", self.p_input)
        pp_form.addRow("Iterations (niter):", self.niter_input)
        pp_layout.addLayout(pp_form)

        # Preset buttons
        preset_row = QHBoxLayout()
        self.btn_save_preset = QPushButton("Simpan Preset")
        self.btn_load_preset = QPushButton("Muat Preset")
        preset_row.addWidget(self.btn_save_preset)
        preset_row.addWidget(self.btn_load_preset)
        pp_layout.addLayout(preset_row)

        # 3) Deteksi puncak
        peak_group = QGroupBox("3. Deteksi Puncak")
        peak_form = QFormLayout(peak_group)
        self.prominence_slider = QSlider(Qt.Orientation.Horizontal)
        self.prominence_slider.setMinimum(1)
        self.prominence_slider.setMaximum(5000)
        self.prominence_slider.setValue(100)
        self.prominence_label = QLabel(f"{self.prominence_slider.value()/10000.0:.4f}")
        prom_row = QHBoxLayout()
        prom_row.addWidget(self.prominence_slider)
        prom_row.addWidget(self.prominence_label)
        self.distance_input = QLineEdit("8")
        self.height_input = QLineEdit(); self.height_input.setPlaceholderText("Opsional, contoh: 0.1")
        self.width_input = QLineEdit(); self.width_input.setPlaceholderText("Opsional, contoh: 20")
        peak_form.addRow("Prominence:", prom_row)
        peak_form.addRow("Distance:", self.distance_input)
        peak_form.addRow("Height:", self.height_input)
        peak_form.addRow("Width:", self.width_input)

        # 4) Interpretasi model
        model_group = QGroupBox("4. Interpretasi Model")
        model_form = QFormLayout(model_group)
        self.threshold_input = QLineEdit("0.6")
        model_form.addRow("Prediction Threshold:", self.threshold_input)

        # 5) Tabs
        self.tabs = QTabWidget()
        predict_tab = QWidget(); predict_layout = QVBoxLayout(predict_tab)
        self.predict_button = QPushButton("Prediksi Elemen"); predict_layout.addWidget(self.predict_button)
        self.predict_tab_index = self.tabs.addTab(predict_tab, "Prediksi Cepat")

        validate_tab = QWidget(); validate_layout = QVBoxLayout(validate_tab)
        self.gt_input = QLineEdit(); self.gt_input.setPlaceholderText("Contoh: Fe, Si, Mg")
        self.validate_button = QPushButton("Validasi Model")
        validate_layout.addWidget(QLabel("Ground Truth Elements:"))
        validate_layout.addWidget(self.gt_input)
        validate_layout.addWidget(self.validate_button)
        self.validate_tab_index = self.tabs.addTab(validate_tab, "Validasi Model")

        # Overlay/Export + Batch
        self.overlay_button = QPushButton("Tambah Overlay Spektrum")
        self.batch_button = QPushButton("Proses Folder (Batch)")
        self.export_button = QPushButton("Ekspor Puncak Berlabel ke XLSX")
        self.export_button.setEnabled(False)

        # Assemble layout
        layout.addWidget(file_group)
        layout.addWidget(pp_group)
        layout.addWidget(peak_group)
        layout.addWidget(model_group)
        layout.addWidget(self.tabs)
        layout.addWidget(self.overlay_button)
        layout.addWidget(self.batch_button)
        layout.addWidget(self.export_button)
        layout.addStretch()
        self.setMaximumWidth(480)

        # Connect signals
        self.prominence_slider.valueChanged.connect(self._on_prominence_changed)
        self.preview_button.clicked.connect(self._on_preview_clicked)
        self.predict_button.clicked.connect(self._on_predict_clicked)
        self.validate_button.clicked.connect(self._on_validate_clicked)
        self.btn_save_preset.clicked.connect(self._on_save_preset)
        self.btn_load_preset.clicked.connect(self._on_load_preset)

    # ----------------- Data helpers -----------------
    def _to_float(self, s: str | None):
        return float(s) if s else None

    def _to_int(self, s: str | None):
        return int(s) if s else None

    def get_parameters(self) -> dict[str, Any]:
        prominence_val = float(self.prominence_slider.value()) / 10000.0
        return {
            "prominence": prominence_val,
            "distance": self._to_int(self.distance_input.text()),
            "height": self._to_float(self.height_input.text()),
            "width": self._to_float(self.width_input.text()),
            "threshold": self._to_float(self.threshold_input.text()),
            "use_raw_resolution": self.raw_resolution_switch.isChecked(),
            "apply_baseline_correction": self.baseline_switch.isChecked(),
            "show_baseline_overlay": self.baseline_overlay_switch.isChecked(),
            "compute_abel": self.abel_switch.isChecked(),
            "lam": self._to_float(self.lam_input.text()),
            "p": self._to_float(self.p_input.text()),
            "niter": self._to_int(self.niter_input.text()),
            "smoothing": self.smooth_switch.isChecked(),
            "sg_window": self._to_int(self.sg_window_input.text()),
            "sg_poly": self._to_int(self.sg_poly_input.text()),
            "normalization": self.norm_combo.currentText(),
            "shift_nm": float(self.shift_spin.value()),
        }

    # ----------------- Signal emitters -----------------
    def _on_prominence_changed(self, value: int):
        self.prominence_label.setText(f"{float(value)/10000.0:.4f}")

    def _on_preview_clicked(self):
        payload = self.get_parameters()
        payload["analysis_mode"] = "preprocess"
        self.previewRequested.emit(payload)

    def _on_predict_clicked(self):
        payload = self.get_parameters()
        payload["analysis_mode"] = "predict"
        self.analysisRequested.emit(payload)

    def _on_validate_clicked(self):
        payload = self.get_parameters()
        payload["analysis_mode"] = "validate"
        payload["ground_truth_elements"] = [
            el.strip() for el in self.gt_input.text().split(",") if el.strip()
        ]
        self.analysisRequested.emit(payload)

    # Convenience setters used by MainWindow
    def set_status(self, text: str):
        self.status_label.setText(text)

    def enable_export(self, on: bool):
        self.export_button.setEnabled(on)

    # ----------------- Preset save/load -----------------
    def _on_save_preset(self):
        params = self.get_parameters()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Simpan Preset Parameter", "", "JSON (*.json)"
        )
        if not filename:
            return
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
            self.status_label.setText(f"Preset disimpan: {filename}")
        except OSError as e:
            self.status_label.setText(f"Gagal menyimpan preset: {e}")

    def _on_load_preset(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Muat Preset Parameter", "", "JSON (*.json)"
        )
        if not filename:
            return
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._apply_parameters_to_ui(data)
                self.status_label.setText(f"Preset dimuat: {filename}")
        except (OSError, json.JSONDecodeError) as e:
            self.status_label.setText(f"Gagal memuat preset: {e}")

    def _apply_parameters_to_ui(self, p: dict[str, Any]):
        # Numeric sliders/spins
        if "shift_nm" in p and isinstance(p["shift_nm"], (int, float)):
            try:
                self.shift_spin.setValue(float(p["shift_nm"]))
            except (TypeError, ValueError):
                pass
        if "prominence" in p and isinstance(p["prominence"], (int, float)):
            try:
                val = max(1, min(5000, int(float(p["prominence"]) * 10000)))
                self.prominence_slider.setValue(val)
            except (TypeError, ValueError):
                pass

        # Text inputs
        def _set_line(line: QLineEdit, key: str):
            if key in p and p[key] is not None:
                line.setText(str(p[key]))

        _set_line(self.distance_input, "distance")
        _set_line(self.height_input, "height")
        _set_line(self.width_input, "width")
        _set_line(self.threshold_input, "threshold")
        _set_line(self.sg_window_input, "sg_window")
        _set_line(self.sg_poly_input, "sg_poly")
        _set_line(self.lam_input, "lam")
        _set_line(self.p_input, "p")
        _set_line(self.niter_input, "niter")

        # Combos and checks
        if "normalization" in p and isinstance(p["normalization"], str):
            idx = self.norm_combo.findText(p["normalization"]) 
            if idx >= 0:
                self.norm_combo.setCurrentIndex(idx)
        if "smoothing" in p:
            self.smooth_switch.setChecked(bool(p["smoothing"]))
        if "apply_baseline_correction" in p:
            self.baseline_switch.setChecked(bool(p["apply_baseline_correction"]))
        if "show_baseline_overlay" in p:
            self.baseline_overlay_switch.setChecked(bool(p["show_baseline_overlay"]))
        if "use_raw_resolution" in p:
            self.raw_resolution_switch.setChecked(bool(p["use_raw_resolution"]))
        if "compute_abel" in p:
            self.abel_switch.setChecked(bool(p["compute_abel"]))
