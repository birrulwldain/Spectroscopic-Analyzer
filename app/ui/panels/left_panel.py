from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QLabel,
        QLineEdit,
        QGroupBox,
        QTabWidget,
        QFormLayout,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QSlider,
    )
else:
    from PySide6 import QtCore as _QtCore  # type: ignore
    from PySide6 import QtWidgets as _QtWidgets  # type: ignore

    Qt = _QtCore.Qt

    QWidget = _QtWidgets.QWidget
    QVBoxLayout = _QtWidgets.QVBoxLayout
    QHBoxLayout = _QtWidgets.QHBoxLayout
    QPushButton = _QtWidgets.QPushButton
    QLabel = _QtWidgets.QLabel
    QLineEdit = _QtWidgets.QLineEdit
    QGroupBox = _QtWidgets.QGroupBox
    QTabWidget = _QtWidgets.QTabWidget
    QFormLayout = _QtWidgets.QFormLayout
    QCheckBox = _QtWidgets.QCheckBox
    QComboBox = _QtWidgets.QComboBox
    QDoubleSpinBox = _QtWidgets.QDoubleSpinBox
    QSlider = _QtWidgets.QSlider


class LeftPanel(QWidget):
    """Left-side controls: file section, Predict/Validate tabs, parameters, overlay/export.

    Exposes widgets for MainWindow to connect signals and a method to gather parameters.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # File group
        file_group = QGroupBox("1. Unggah Data")
        file_layout = QVBoxLayout(file_group)
        self.file_button = QPushButton("Pilih File .asc")
        self.status_label = QLabel("Silakan pilih file.")
        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.status_label)

        # Tabs
        self.tabs = QTabWidget()

        # Predict tab
        predict_tab = QWidget()
        predict_layout = QVBoxLayout(predict_tab)
        self.preprocess_button = QPushButton("1) Preprocessing")
        self.preprocess_button.setToolTip(
            "Jalankan preprocessing + deteksi puncak secara interaktif tanpa pemetaan elemen"
        )
        self.predict_button = QPushButton("2) Prediksi Elemen")
        self.predict_button.setToolTip(
            "Lakukan pemetaan elemen berdasarkan puncak yang terdeteksi"
        )
        predict_layout.addWidget(self.preprocess_button)
        predict_layout.addWidget(self.predict_button)
        self.predict_tab_index = self.tabs.addTab(predict_tab, "Prediksi Cepat")

        # Validate tab
        validate_tab = QWidget()
        validate_layout = QVBoxLayout(validate_tab)
        self.gt_input = QLineEdit()
        self.gt_input.setPlaceholderText("Contoh: Fe, Si, Mg")
        self.validate_button = QPushButton("Jalankan Validasi")
        validate_layout.addWidget(QLabel("Ground Truth Elements:"))
        validate_layout.addWidget(self.gt_input)
        validate_layout.addWidget(self.validate_button)
        self.validate_tab_index = self.tabs.addTab(validate_tab, "Validasi Model")

        # Parameters
        param_group = QGroupBox("Parameter Opsional")
        form = QFormLayout(param_group)

        self.prominence_slider = QSlider(Qt.Orientation.Horizontal)
        self.prominence_slider.setMinimum(1)
        self.prominence_slider.setMaximum(5000)
        self.prominence_slider.setValue(100)
        self.prominence_label = QLabel("")
        self.prominence_label.setText(f"{self.prominence_slider.value()/10000.0:.4f}")
        prom_row = QHBoxLayout()
        prom_row.addWidget(self.prominence_slider)
        prom_row.addWidget(self.prominence_label)

        self.distance_input = QLineEdit("8")
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("Opsional, contoh: 0.1")
        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText("Opsional, contoh: 20")
        self.threshold_input = QLineEdit("0.6")

        form.addRow("Prominence:", prom_row)
        form.addRow("Distance:", self.distance_input)
        form.addRow("Height:", self.height_input)
        form.addRow("Width:", self.width_input)
        form.addRow("Prediction Threshold:", self.threshold_input)

        self.baseline_switch = QCheckBox("Aktifkan Koreksi Baseline")
        self.baseline_overlay_switch = QCheckBox("Tampilkan Overlay Baseline")
        self.raw_resolution_switch = QCheckBox("Gunakan resolusi asli (tanpa resampling)")
        self.raw_resolution_switch.setChecked(True)
        self.abel_switch = QCheckBox("Hitung Inversi Abel (basex)")
        self.lam_input = QLineEdit("100000")
        self.p_input = QLineEdit("0.01")
        self.niter_input = QLineEdit("10")

        form.addRow(self.raw_resolution_switch)
        form.addRow(self.baseline_switch)
        form.addRow(self.baseline_overlay_switch)
        form.addRow(self.abel_switch)
        form.addRow("Lambda (lam):", self.lam_input)
        form.addRow("Asymmetry (p):", self.p_input)
        form.addRow("Iterations (niter):", self.niter_input)

        self.smooth_switch = QCheckBox("Smoothing (Savitzky–Golay)")
        self.sg_window_input = QLineEdit("11")
        self.sg_poly_input = QLineEdit("3")
        form.addRow(self.smooth_switch)
        form.addRow("SG Window (odd):", self.sg_window_input)
        form.addRow("SG Polyorder:", self.sg_poly_input)

        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["None", "Max", "Area"])
        form.addRow("Normalization:", self.norm_combo)

        self.shift_spin = QDoubleSpinBox()
        self.shift_spin.setRange(-10.0, 10.0)
        self.shift_spin.setDecimals(3)
        self.shift_spin.setSingleStep(0.01)
        self.shift_spin.setValue(0.0)
        form.addRow("Wavelength shift (nm):", self.shift_spin)

        # Overlay/Export
        self.overlay_button = QPushButton("Tambah Overlay Spektrum")
        self.export_button = QPushButton("Ekspor Puncak Berlabel ke XLSX")
        self.export_button.setEnabled(False)

        # Assemble
        layout.addWidget(file_group)
        layout.addWidget(self.tabs)
        layout.addWidget(param_group)
        layout.addWidget(self.overlay_button)
        layout.addWidget(self.export_button)
        layout.addStretch()
        self.setMaximumWidth(450)

    # ---- API for MainWindow ----
    def get_parameters(self) -> dict[str, Any]:
        def to_float(s: str | None):
            return float(s) if s else None

        def to_int(s: str | None):
            return int(s) if s else None

        prominence_val = float(self.prominence_slider.value()) / 10000.0
        return {
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

    def set_status(self, text: str):
        self.status_label.setText(text)

    def enable_export(self, on: bool):
        self.export_button.setEnabled(on)
