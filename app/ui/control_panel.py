from __future__ import annotations
from typing import Any, TYPE_CHECKING
import json

from app.ui.panel_style import CONTROL_PANEL_STYLE

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
        QSizePolicy,
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
    QSizePolicy = _QtWidgets.QSizePolicy


class ControlPanel(QWidget):
    previewRequested = Signal(dict)
    analysisRequested = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Set size policy untuk panel kontrol agar bisa memperluas secara vertikal
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        self.setStyleSheet(CONTROL_PANEL_STYLE)  # Terapkan stylesheet global
        self._build_ui()

    def _build_ui(self) -> None:
        # Use horizontal layout for control panel
        layout = QHBoxLayout(self)
        
        # Left column: File selection
        left_column = QVBoxLayout()
        left_column.setSpacing(8)  # Kurangi spacing untuk membuat layout lebih kompak
        
        # 1) File group + prominent Preview button dengan styling yang lebih baik
        file_group = QGroupBox("1. Unggah Data")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(6)  # Kurangi spacing untuk membuat layout lebih kompak
        
        self.file_button = QPushButton("Pilih File .asc")
        self.file_button.setStyleSheet("""
            QPushButton {
                background-color: #f8f9fa;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)
        
        self.status_label = QLabel("Silakan pilih file.")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 5px;
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 3px;
                color: #495057;
            }
        """)
        
        self.preview_button = QPushButton("Terapkan & Pratinjau")
        self.preview_button.setMinimumHeight(40)  # Make button more prominent
        self.preview_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0069d9;
            }
            QPushButton:pressed {
                background-color: #0062cc;
            }
        """)
        
        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.status_label)
        file_layout.addSpacing(5)
        file_layout.addWidget(self.preview_button)
        
        left_column.addWidget(file_group)
        layout.addLayout(left_column, 25)  # 25% width

        # 2) Pra-pemrosesan Sinyal dengan styling yang memastikan ukuran yang tepat
        pp_group = QGroupBox("2. Pra-pemrosesan Sinyal")
        pp_layout = QVBoxLayout(pp_group)
        pp_layout.setSpacing(10)
        pp_layout.setContentsMargins(10, 15, 10, 10)
        
        # Setel style sheet umum untuk semua group boxes
        groupbox_style = """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """
        pp_group.setStyleSheet(groupbox_style)
        
        # Gunakan grup-grup kecil untuk mengkategorikan pengaturan
        basic_group = QGroupBox("Pengaturan Dasar")
        basic_group.setStyleSheet(groupbox_style)
        basic_form = QFormLayout(basic_group)
        basic_form.setContentsMargins(10, 15, 10, 10)
        
        self.shift_spin = QDoubleSpinBox()
        self.shift_spin.setRange(-10.0, 10.0)
        self.shift_spin.setDecimals(3)
        self.shift_spin.setSingleStep(0.01)
        self.shift_spin.setValue(0.0)
        self.shift_spin.setStyleSheet("QDoubleSpinBox { padding: 3px; }")

        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["None", "Max", "Area"])
        self.norm_combo.setStyleSheet("QComboBox { padding: 3px; }")
        
        basic_form.addRow("Wavelength shift (nm):", self.shift_spin)
        basic_form.addRow("Normalization:", self.norm_combo)
        pp_layout.addWidget(basic_group)
        
        # Grup Smoothing
        smooth_group = QGroupBox("Smoothing")
        smooth_group.setStyleSheet(groupbox_style)
        smooth_form = QFormLayout(smooth_group)
        smooth_form.setContentsMargins(10, 15, 10, 10)
        
        self.smooth_switch = QCheckBox("Aktifkan Smoothing (Savitzky–Golay)")
        self.sg_window_input = QLineEdit("11")
        self.sg_poly_input = QLineEdit("3")
        self.sg_window_input.setStyleSheet("QLineEdit { padding: 3px; }")
        self.sg_poly_input.setStyleSheet("QLineEdit { padding: 3px; }")
        
        smooth_form.addRow(self.smooth_switch)
        smooth_form.addRow("SG Window (odd):", self.sg_window_input)
        smooth_form.addRow("SG Polyorder:", self.sg_poly_input)
        pp_layout.addWidget(smooth_group)
        
        # Grup Baseline dan Resolusi
        baseline_group = QGroupBox("Baseline dan Resolusi")
        baseline_group.setStyleSheet(groupbox_style)
        baseline_form = QFormLayout(baseline_group)
        baseline_form.setContentsMargins(10, 15, 10, 10)
        
        self.baseline_switch = QCheckBox("Aktifkan Koreksi Baseline")
        self.baseline_overlay_switch = QCheckBox("Tampilkan Overlay Baseline")
        self.raw_resolution_switch = QCheckBox("Gunakan resolusi asli (tanpa resampling)")
        self.raw_resolution_switch.setChecked(True)
        
        # Parameter koreksi baseline (dipindah dari grup Abel)
        self.lam_input = QLineEdit("100000")
        self.lam_input.setStyleSheet("QLineEdit { padding: 3px; }")
        self.p_input = QLineEdit("0.01") 
        self.p_input.setStyleSheet("QLineEdit { padding: 3px; }")
        self.niter_input = QLineEdit("10")
        self.niter_input.setStyleSheet("QLineEdit { padding: 3px; }")
        
        baseline_form.addRow(self.baseline_switch)
        baseline_form.addRow(self.baseline_overlay_switch)
        baseline_form.addRow("Lambda:", self.lam_input)
        baseline_form.addRow("Asymmetry (p):", self.p_input)
        baseline_form.addRow("Iterations:", self.niter_input)
        baseline_form.addRow(self.raw_resolution_switch)
        pp_layout.addWidget(baseline_group)

        # Tetap membuat variabel abel_switch sebagai atribut tetapi tidak ditampilkan di UI
        # Agar tidak perlu mengubah logika di model.py atau processing.py
        self.abel_switch = QCheckBox("Hitung Inversi Abel (basex)")
        self.abel_switch.setVisible(False)

        # Preset buttons dengan styling yang lebih baik
        preset_group = QGroupBox("Preset")
        preset_group.setStyleSheet(groupbox_style)
        preset_layout = QHBoxLayout(preset_group)
        preset_layout.setContentsMargins(10, 10, 10, 10)
        
        self.btn_save_preset = QPushButton("Simpan Preset")
        self.btn_load_preset = QPushButton("Muat Preset")
        self.btn_save_preset.setStyleSheet("QPushButton { padding: 5px 10px; }")
        self.btn_load_preset.setStyleSheet("QPushButton { padding: 5px 10px; }")
        preset_layout.addWidget(self.btn_save_preset)
        preset_layout.addWidget(self.btn_load_preset)
        pp_layout.addWidget(preset_group)

        # 3) Deteksi puncak dengan styling yang lebih baik
        peak_group = QGroupBox("3. Deteksi Puncak")
        peak_group.setStyleSheet(groupbox_style)
        peak_layout = QVBoxLayout(peak_group)
        peak_layout.setSpacing(10)
        peak_layout.setContentsMargins(10, 15, 10, 10)
        
        # Prominence dengan input field
        prominence_group = QGroupBox("Prominence")
        prominence_group.setStyleSheet(groupbox_style)
        prominence_layout = QVBoxLayout(prominence_group)
        prominence_layout.setContentsMargins(10, 15, 10, 10)
        
        # Tambahkan label untuk prominence
        prominence_label = QLabel("Nilai minimum prominence untuk deteksi puncak:")
        prominence_label.setStyleSheet("QLabel { color: #495057; font-size: 12px; margin-bottom: 5px; }")
        prominence_layout.addWidget(prominence_label)
        
        self.prominence_input = QLineEdit("0.01")
        self.prominence_input.setStyleSheet("QLineEdit { padding: 3px; }")
        self.prominence_input.setPlaceholderText("Contoh: 0.01")
        
        prominence_layout.addWidget(self.prominence_input)
        peak_layout.addWidget(prominence_group)
        
        # Parameter deteksi puncak lainnya
        params_group = QGroupBox("Parameter Lainnya")
        params_group.setStyleSheet(groupbox_style)
        params_form = QFormLayout(params_group)
        params_form.setContentsMargins(10, 15, 10, 10)
        
        self.distance_input = QLineEdit("3")
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("Opsional, contoh: 0.1")
        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText("Opsional, contoh: 20")
        
        self.distance_input.setStyleSheet("QLineEdit { padding: 3px; }")
        self.height_input.setStyleSheet("QLineEdit { padding: 3px; }")
        self.width_input.setStyleSheet("QLineEdit { padding: 3px; }")
        
        params_form.addRow("Distance:", self.distance_input)
        params_form.addRow("Height:", self.height_input)
        params_form.addRow("Width:", self.width_input)
        peak_layout.addWidget(params_group)

        # 4) Interpretasi model dengan styling yang lebih baik
        model_group = QGroupBox("4. Interpretasi Model")
        model_group.setStyleSheet(groupbox_style)
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(10)
        model_layout.setContentsMargins(10, 15, 10, 10)
        
        # Threshold panel
        threshold_group = QGroupBox("Threshold Setting")
        threshold_group.setStyleSheet(groupbox_style)
        threshold_form = QFormLayout(threshold_group)
        threshold_form.setContentsMargins(10, 15, 10, 10)
        
        self.threshold_input = QLineEdit("0.1")
        self.threshold_input.setStyleSheet("QLineEdit { padding: 3px; }")
        self.threshold_input.setPlaceholderText("Threshold intensitas minimum")
        threshold_form.addRow("Intensity Threshold:", self.threshold_input)
        
        self.prediction_threshold_input = QLineEdit("0.7")
        self.prediction_threshold_input.setStyleSheet("QLineEdit { padding: 3px; font-weight: bold; }")
        self.prediction_threshold_input.setPlaceholderText("Threshold probabilitas sigmoid (0-1)")
        threshold_form.addRow("Prediction Threshold:", self.prediction_threshold_input)
        
        model_layout.addWidget(threshold_group)

        # 5) Tabs dengan styling yang lebih baik
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background: white;
                border-radius: 3px;
            }
            QTabBar::tab {
                background: #e6e6e6;
                padding: 6px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #0066cc;
                color: white;
            }
        """)
        
        # Prediksi Tab
        predict_tab = QWidget()
        predict_layout = QVBoxLayout(predict_tab)
        predict_layout.setContentsMargins(10, 15, 10, 10)
        
        predict_desc = QLabel("Klik tombol di bawah untuk menjalankan prediksi cepat elemen berdasarkan spektrum.")
        predict_desc.setWordWrap(True)
        predict_desc.setStyleSheet("QLabel { color: #333333; }")
        
        self.predict_button = QPushButton("Prediksi Elemen")
        self.predict_button.setMinimumHeight(40)
        self.predict_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        predict_layout.addWidget(predict_desc)
        predict_layout.addSpacing(10)
        predict_layout.addWidget(self.predict_button)
        predict_layout.addStretch(1)
        
        self.predict_tab_index = self.tabs.addTab(predict_tab, "Prediksi Cepat")

        # Validasi Tab
        validate_tab = QWidget()
        validate_layout = QVBoxLayout(validate_tab)
        validate_layout.setContentsMargins(10, 15, 10, 10)
        
        validate_desc = QLabel("Masukkan elemen yang diketahui ada dalam sampel untuk memvalidasi kinerja model.")
        validate_desc.setWordWrap(True)
        validate_desc.setStyleSheet("QLabel { color: #333333; }")
        
        gt_label = QLabel("Ground Truth Elements:")
        gt_label.setStyleSheet("QLabel { font-weight: bold; }")
        
        self.gt_input = QLineEdit()
        self.gt_input.setPlaceholderText("Contoh: Fe, Si, Mg")
        self.gt_input.setStyleSheet("QLineEdit { padding: 5px; }")
        
        self.validate_button = QPushButton("Validasi Model")
        self.validate_button.setMinimumHeight(40)
        self.validate_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        
        validate_layout.addWidget(validate_desc)
        validate_layout.addSpacing(10)
        validate_layout.addWidget(gt_label)
        validate_layout.addWidget(self.gt_input)
        validate_layout.addSpacing(10)
        validate_layout.addWidget(self.validate_button)
        validate_layout.addStretch(1)
        
        self.validate_tab_index = self.tabs.addTab(validate_tab, "Validasi Model")

        # Overlay/Export + Batch dengan styling yang lebih baik
        actions_group = QGroupBox("Aksi Tambahan")
        actions_group.setStyleSheet(groupbox_style)
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setSpacing(10)
        actions_layout.setContentsMargins(10, 15, 10, 10)
        
        self.overlay_button = QPushButton("Tambah Overlay Spektrum")
        self.batch_button = QPushButton("Proses Folder (Batch)")
        self.export_button = QPushButton("Ekspor Puncak Berlabel ke XLSX")
        
        # Styling tombol-tombol
        action_button_style = """
            QPushButton {
                padding: 8px;
                border-radius: 4px;
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #aaaaaa;
            }
        """
        self.overlay_button.setStyleSheet(action_button_style)
        self.batch_button.setStyleSheet(action_button_style)
        self.export_button.setStyleSheet(action_button_style)
        self.export_button.setEnabled(False)
        
        actions_layout.addWidget(self.overlay_button)
        actions_layout.addWidget(self.batch_button)
        actions_layout.addWidget(self.export_button)

        # Middle column
        middle_column = QVBoxLayout()
        middle_column.addWidget(pp_group)
        middle_column.addWidget(peak_group)
        layout.addLayout(middle_column, 35)  # 35% width
        
        # Right column
        right_column = QVBoxLayout()
        right_column.addWidget(model_group)
        right_column.addWidget(self.tabs)
        right_column.addWidget(actions_group)
        layout.addLayout(right_column, 40)  # 40% width

        # Connect signals
        self.prominence_input.textChanged.connect(self._on_prominence_changed)
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
        try:
            prominence_val = float(self.prominence_input.text())
        except ValueError:
            prominence_val = 0.01  # default value
        return {
            "prominence": prominence_val,
            "distance": self._to_int(self.distance_input.text()),
            "height": self._to_float(self.height_input.text()),
            "width": self._to_float(self.width_input.text()),
            "threshold": self._to_float(self.threshold_input.text()),
            "prediction_threshold": self._to_float(self.prediction_threshold_input.text()),
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
    def _on_prominence_changed(self, text: str):
        # No need to update label since we removed it
        pass

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
                self.prominence_input.setText(str(float(p["prominence"])))
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
        _set_line(self.prediction_threshold_input, "prediction_threshold")
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
