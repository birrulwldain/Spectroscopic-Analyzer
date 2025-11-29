from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectroscopic Analyzer")
        self.resize(1200, 700)
        self._setup_ui()

    def _setup_ui(self):
        # Central widget and main layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # MAIN LAYOUT: Grid 2x2 (4 kuadran rata)
        main_layout = QtWidgets.QGridLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # ========== KUADRAN KIRI ATAS: PLOT UTAMA ==========
        left_top_panel = QtWidgets.QWidget()
        left_top_layout = QtWidgets.QVBoxLayout(left_top_panel)
        left_top_layout.setContentsMargins(0, 0, 0, 0)

        # Top-left: full spectrum label
        left_top_label = QtWidgets.QLabel("Full Spectrum - Drag Region Selector")
        left_top_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #4a90e2; color: white;")
        left_top_layout.addWidget(left_top_label)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Intensity', **{'font-size': '11pt'})
        self.plot_widget.setLabel('bottom', 'Wavelength (nm)', **{'font-size': '11pt'})
        self.plot_widget.showGrid(x=True, y=True)
        left_top_layout.addWidget(self.plot_widget)

        # ========== KUADRAN KANAN ATAS: PLOT PREVIEW ==========
        right_top_panel = QtWidgets.QWidget()
        right_top_layout = QtWidgets.QVBoxLayout(right_top_panel)
        right_top_layout.setContentsMargins(0, 0, 0, 0)

        # Header dengan tombol ekspor
        right_top_header_layout = QtWidgets.QHBoxLayout()

        # Top-right: preview label
        right_top_label = QtWidgets.QLabel("Preview: Element Labels (Selected Region)")
        right_top_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #50c878; color: white;")
        right_top_header_layout.addWidget(right_top_label, 1)

        # Export publication-quality plot
        btn_export_plot = QtWidgets.QPushButton("Export Scientific Plot")
        btn_export_plot.setStyleSheet("font-weight: bold; padding: 3px 8px; background-color: #FFD700; color: black;")
        btn_export_plot.setToolTip("Export the current plot as a publication-ready scientific figure")
        right_top_header_layout.addWidget(btn_export_plot)

        right_top_layout.addLayout(right_top_header_layout)

        self.detail_plot_widget = pg.PlotWidget()
        self.detail_plot_widget.setLabel('left', 'Intensity', **{'font-size': '11pt'})
        self.detail_plot_widget.setLabel('bottom', 'Wavelength (nm)', **{'font-size': '11pt'})
        self.detail_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        right_top_layout.addWidget(self.detail_plot_widget)

        # ========== KUADRAN KIRI BAWAH: INPUT PARAMETER ==========
        left_bottom_panel = QtWidgets.QWidget()
        left_bottom_layout = QtWidgets.QVBoxLayout(left_bottom_panel)
        left_bottom_layout.setContentsMargins(0, 0, 0, 0)

        # Header
        param_header = QtWidgets.QLabel("Control Panel & Parameters")
        param_header.setStyleSheet("font-weight: bold; padding: 5px; background-color: #ff6b6b; color: white;")
        left_bottom_layout.addWidget(param_header)

        # Scroll area untuk parameters
        param_scroll = QtWidgets.QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        param_container = QtWidgets.QWidget()
        param_container_layout = QtWidgets.QVBoxLayout(param_container)

        # File operations group
        file_group = QtWidgets.QGroupBox("File Operations")
        file_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        file_layout = QtWidgets.QVBoxLayout(file_group)
        btn_load_folder = QtWidgets.QPushButton("Load Folder")
        btn_load = QtWidgets.QPushButton("Load File")
        btn_save = QtWidgets.QPushButton("Save")
        btn_export = QtWidgets.QPushButton("Export")
        btn_preprocess = QtWidgets.QPushButton("Preprocess")
        btn_predict = QtWidgets.QPushButton("Predict")

        for btn in [btn_load_folder, btn_load, btn_save, btn_export, btn_preprocess, btn_predict]:
            btn.setMinimumHeight(35)

        file_layout.addWidget(btn_load_folder)
        file_layout.addWidget(btn_load)
        file_layout.addWidget(btn_save)
        file_layout.addWidget(btn_export)
        file_layout.addWidget(btn_preprocess)
        file_layout.addWidget(btn_predict)
        param_container_layout.addWidget(file_group)

        # Baseline Correction group
        baseline_group = QtWidgets.QGroupBox("Baseline Correction (ALS)")
        baseline_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        baseline_layout = QtWidgets.QVBoxLayout(baseline_group)

        baseline_label = QtWidgets.QLabel("ALS Lambda:")
        baseline_input = QtWidgets.QLineEdit("100000")
        baseline_input.setPlaceholderText("Regularization parameter")

        intensity_label = QtWidgets.QLabel("Target Max Intensity:")
        intensity_input = QtWidgets.QLineEdit("0.8")
        intensity_input.setPlaceholderText("0.0 - 1.0")

        anchor_label = QtWidgets.QLabel("ALS p:")
        anchor_input = QtWidgets.QLineEdit("0.001")
        anchor_input.setPlaceholderText("Asymmetry parameter (0-1)")

        max_iter_label = QtWidgets.QLabel("ALS Max Iterations:")
        max_iter_input = QtWidgets.QLineEdit("10")
        max_iter_input.setPlaceholderText("Number of iterations")

        baseline_layout.addWidget(baseline_label)
        baseline_layout.addWidget(baseline_input)
        baseline_layout.addWidget(intensity_label)
        baseline_layout.addWidget(intensity_input)
        baseline_layout.addWidget(anchor_label)
        baseline_layout.addWidget(anchor_input)
        baseline_layout.addWidget(max_iter_label)
        baseline_layout.addWidget(max_iter_input)
        param_container_layout.addWidget(baseline_group)

        # Prediction Parameters group
        prediction_group = QtWidgets.QGroupBox("Prediction Parameters")
        prediction_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        prediction_layout = QtWidgets.QVBoxLayout(prediction_group)

        threshold_label = QtWidgets.QLabel("Detection Threshold:")
        threshold_input = QtWidgets.QLineEdit("0.4")
        threshold_input.setPlaceholderText("0.0 - 1.0")

        min_height_label = QtWidgets.QLabel("Min Peak Height:")
        min_height_input = QtWidgets.QLineEdit("0.1")
        min_height_input.setPlaceholderText("Minimum probability for peak")

        min_width_label = QtWidgets.QLabel("Min Peak Width:")
        min_width_input = QtWidgets.QLineEdit("1")
        min_width_input.setPlaceholderText("Minimum width in data points")

        max_peaks_label = QtWidgets.QLabel("Max Peaks per Element:")
        max_peaks_input = QtWidgets.QLineEdit("3")
        max_peaks_input.setPlaceholderText("Maximum number of peaks")

        prominence_label = QtWidgets.QLabel("Peak Prominence:")
        prominence_input = QtWidgets.QLineEdit("0.05")
        prominence_input.setPlaceholderText("Minimum prominence")

        min_distance_label = QtWidgets.QLabel("Min Peak Distance:")
        min_distance_input = QtWidgets.QLineEdit("5")
        min_distance_input.setPlaceholderText("Minimum distance between peaks")

        self.threshold_input = threshold_input
        self.min_height_input = min_height_input
        self.min_width_input = min_width_input
        self.max_peaks_input = max_peaks_input
        self.prominence_input = prominence_input
        self.min_distance_input = min_distance_input

        prediction_layout.addWidget(threshold_label)
        prediction_layout.addWidget(threshold_input)
        prediction_layout.addWidget(min_height_label)
        prediction_layout.addWidget(min_height_input)
        prediction_layout.addWidget(min_width_label)
        prediction_layout.addWidget(min_width_input)
        prediction_layout.addWidget(max_peaks_label)
        prediction_layout.addWidget(max_peaks_input)
        prediction_layout.addWidget(prominence_label)
        prediction_layout.addWidget(prominence_input)
        prediction_layout.addWidget(min_distance_label)
        prediction_layout.addWidget(min_distance_input)
        param_container_layout.addWidget(prediction_group)

        # Wavelength Range group
        wavelength_group = QtWidgets.QGroupBox("Wavelength Range")
        wavelength_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        wavelength_layout = QtWidgets.QVBoxLayout(wavelength_group)

        min_wavelength_label = QtWidgets.QLabel("Min Wavelength (nm):")
        min_wavelength_input = QtWidgets.QLineEdit("200")
        min_wavelength_input.setPlaceholderText("Minimum wavelength")

        max_wavelength_label = QtWidgets.QLabel("Max Wavelength (nm):")
        max_wavelength_input = QtWidgets.QLineEdit("800")
        max_wavelength_input.setPlaceholderText("Maximum wavelength")

        self.min_wavelength_input = min_wavelength_input
        self.max_wavelength_input = max_wavelength_input

        wavelength_layout.addWidget(min_wavelength_label)
        wavelength_layout.addWidget(min_wavelength_input)
        wavelength_layout.addWidget(max_wavelength_label)
        wavelength_layout.addWidget(max_wavelength_input)
        param_container_layout.addWidget(wavelength_group)

        # Results group
        results_group = QtWidgets.QGroupBox("Results")
        results_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        results_layout = QtWidgets.QVBoxLayout(results_group)
        self.results_label = QtWidgets.QLabel("Results will appear here after prediction.")
        results_layout.addWidget(self.results_label)
        param_container_layout.addWidget(results_group)

        param_container_layout.addStretch()
        param_scroll.setWidget(param_container)
        left_bottom_layout.addWidget(param_scroll)

        # ========== KUADRAN KANAN BAWAH: FILE LIST & LOG ==========
        right_bottom_panel = QtWidgets.QWidget()
        right_bottom_layout = QtWidgets.QVBoxLayout(right_bottom_panel)
        right_bottom_layout.setContentsMargins(0, 0, 0, 0)

        # Header
        file_log_header = QtWidgets.QLabel("Files & Log")
        file_log_header.setStyleSheet("font-weight: bold; padding: 5px; background-color: #9b59b6; color: white;")
        right_bottom_layout.addWidget(file_log_header)

        # File list table
        file_group_bottom = QtWidgets.QGroupBox("Loaded Files")
        file_group_bottom.setStyleSheet("QGroupBox { font-weight: bold; }")
        file_table_layout = QtWidgets.QVBoxLayout(file_group_bottom)

        self.data_table = QtWidgets.QTableWidget(0, 3)
        self.data_table.setHorizontalHeaderLabels(["Filename", "Intensity", "Status"])
        self.data_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.data_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.data_table.horizontalHeader().setStretchLastSection(True)
        file_table_layout.addWidget(self.data_table)
        right_bottom_layout.addWidget(file_group_bottom, stretch=1)

        # Log panel
        log_group = QtWidgets.QGroupBox("Activity Log")
        log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        log_layout = QtWidgets.QVBoxLayout(log_group)

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("background-color: #2c3e50; color: #ecf0f1; font-family: 'Courier New';")
        log_layout.addWidget(self.log_text)
        right_bottom_layout.addWidget(log_group, stretch=1)

        # ========== ASSIGN KUADRAN KE GRID LAYOUT ==========
        # Grid Layout: (row, col, rowSpan, colSpan)
        main_layout.addWidget(left_top_panel, 0, 0)      # Kiri Atas
        main_layout.addWidget(right_top_panel, 0, 1)     # Kanan Atas
        main_layout.addWidget(left_bottom_panel, 1, 0)   # Kiri Bawah
        main_layout.addWidget(right_bottom_panel, 1, 1)  # Kanan Bawah

        # Set proporsi yang sama untuk semua kuadran
        main_layout.setRowStretch(0, 1)
        main_layout.setRowStretch(1, 1)
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 1)

        # Connect button signals
        btn_load.clicked.connect(self.load_file)
        btn_load_folder.clicked.connect(self.load_folder)
        btn_preprocess.clicked.connect(self.preprocess_current_file)
        btn_predict.clicked.connect(self.predict_current_file)
        btn_export_plot.clicked.connect(self.export_publication_plot)
        self.data_table.cellClicked.connect(self.on_file_selected)

        # Menu bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction("Open")
        file_menu.addAction("Save")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        view_menu = menu_bar.addMenu("View")
        analysis_menu = menu_bar.addMenu("Analysis")
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("About")

        # Tool bar
        tool_bar = self.addToolBar("Main Toolbar")
        tool_bar.addAction("Load")
        tool_bar.addAction("Save")
        tool_bar.addAction("Start")
        tool_bar.addAction("Stop")
        self.min_wavelength_input.editingFinished.connect(self.update_region_from_inputs)
        self.max_wavelength_input.editingFinished.connect(self.update_region_from_inputs)
        tool_bar.addAction("Pan")
        tool_bar.addAction("Reset")
        tool_bar.addSeparator()
        tool_bar.addAction("Export Plot")

        # Status bar
        self.statusBar().showMessage("Ready | Device: Disconnected | Mode: Idle")

        # Initialize variables
        self.loaded_folder = None
        self.file_list = []
        self.current_file = None
        self.preprocessed_spectrum = None

        # Parameter inputs
        self.threshold_input = threshold_input
        self.baseline_input = baseline_input
        self.intensity_input = intensity_input
        self.anchor_input = anchor_input
        self.max_iter_input = max_iter_input
        self.min_height_input = min_height_input
        self.min_width_input = min_width_input
        self.max_peaks_input = max_peaks_input
        self.prominence_input = prominence_input
        self.min_distance_input = min_distance_input

        # Assets cache
        self._assets_loaded = False
        self._cached_model = None
        self._cached_element_map = None
        self._cached_target_wavelengths = None


    # Helper: ensure assets are loaded once and valid
    def ensure_assets_loaded(self) -> bool:
        """Load and cache model/assets once. Returns True on success."""
        if self._assets_loaded:
            return True
        try:
            from app.model import load_assets
            assets = load_assets()
            if not isinstance(assets, tuple) or len(assets) != 3:
                self.log_text.append("load_assets() returned unexpected value — expected (model, element_map, wavelengths)")
                return False
            model, element_map, target_wavelengths = assets
            # Basic validations
            if not hasattr(model, '__call__'):
                self.log_text.append(f"Loaded model object is not callable: {type(model)}")
                return False
            if not isinstance(element_map, dict):
                self.log_text.append(f"element_map has unexpected type: {type(element_map)}")
                return False
            import numpy as _np
            if not isinstance(target_wavelengths, _np.ndarray):
                self.log_text.append(f"target_wavelengths has unexpected type: {type(target_wavelengths)}")
                return False

            self._cached_model = model
            self._cached_element_map = element_map
            self._cached_target_wavelengths = target_wavelengths
            self._assets_loaded = True
            self.log_text.append("Aset model berhasil dimuat dan dicache.")
            return True
        except Exception as e:
            import traceback
            self.log_text.append(f"Error loading assets: {e}")
            self.log_text.append(traceback.format_exc())
            return False

    def load_file(self):
        file_dialog = QtWidgets.QFileDialog(self, "Open Data File", "", "Data Files (*.asc *.txt *.csv);;All Files (*)")
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                self.log_text.append(f"Loaded file: {file_path}\n---\n{content[:1000]}\n---")
            except Exception as e:
                self.log_text.append(f"Error loading file: {e}")

    def load_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder with ASC Files")
        if folder:
            import os
            self.loaded_folder = folder
            self.file_list = [f for f in os.listdir(folder) if f.lower().endswith('.asc')]
            self.data_table.setRowCount(len(self.file_list))
            for i, fname in enumerate(self.file_list):
                self.data_table.setItem(i, 0, QtWidgets.QTableWidgetItem(fname))
                self.data_table.setItem(i, 1, QtWidgets.QTableWidgetItem("-"))
                self.data_table.setItem(i, 2, QtWidgets.QTableWidgetItem("-"))
            self.log_text.append(f"Loaded folder: {folder} ({len(self.file_list)} .asc files)")
            if not self.file_list:
                self.log_text.append("No .asc files found in the selected folder.")

    def on_file_selected(self, row, col):
        if not self.file_list or row >= len(self.file_list):
            self.log_text.append("No file selected or out of range.")
            return
        fname = self.file_list[row]
        self.current_file = fname
        import os
        fpath = os.path.join(self.loaded_folder, fname)
        try:
            with open(fpath, 'r') as f:
                lines = f.readlines()
            # Parse ASC: cari dua kolom float
            import numpy as np
            data = []
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            wl, inten = float(parts[0]), float(parts[1])
                            data.append((wl, inten))
                        except Exception:
                            continue
            if data:
                arr = np.array(data)
                self.plot_widget.clear()
                self.plot_widget.plot(arr[:,0], arr[:,1], pen='b')
                # Update kolom intensity (max) di tabel
                max_inten = str(np.max(arr[:,1]))
                self.data_table.setItem(row, 1, QtWidgets.QTableWidgetItem(max_inten))
                self.log_text.append(f"Plotted: {fname}")
            else:
                self.plot_widget.clear()
                self.data_table.setItem(row, 1, QtWidgets.QTableWidgetItem("-"))
                self.log_text.append(f"No valid data in: {fname}")
        except Exception as e:
            self.plot_widget.clear()
            self.log_text.append(f"Error reading {fname}: {e}")

    def preprocess_current_file(self):
        if not self.current_file or not self.loaded_folder:
            self.log_text.append("No file selected for preprocessing.")
            return
        import os
        import numpy as np
        from app.processing import prepare_asc_data
        try:
            ok = self.ensure_assets_loaded()
            if not ok:
                self.log_text.append("Cannot preprocess because assets failed to load.")
                return
            model = self._cached_model
            element_map = self._cached_element_map
            target_wavelengths = self._cached_target_wavelengths
        except Exception as e:
            self.log_text.append(f"Error loading assets: {e}")
            return

        fpath = os.path.join(self.loaded_folder, self.current_file)
        try:
            with open(fpath, 'r') as f:
                content = f.read()
            # Get parameters from inputs
            try:
                als_lambda = float(self.baseline_input.text())
                target_max_intensity = float(self.intensity_input.text())
                als_p = float(self.anchor_input.text())
                als_max_iter = int(self.max_iter_input.text())
            except ValueError:
                self.log_text.append("Invalid input for ALS parameters. Using defaults.")
                als_lambda = 1e5
                target_max_intensity = 0.8
                als_p = 0.001
                als_max_iter = 10
            self.preprocessed_spectrum = prepare_asc_data(content, target_wavelengths, target_max_intensity, als_lambda, als_p, als_max_iter)
            self.plot_widget.clear()
            self.plot_widget.plot(target_wavelengths, self.preprocessed_spectrum, pen='g')
            # Update note di tabel
            row = self.file_list.index(self.current_file)
            self.data_table.setItem(row, 2, QtWidgets.QTableWidgetItem("Preprocessed"))
            self.log_text.append(f"Preprocessed: {self.current_file}")
        except Exception as e:
            self.log_text.append(f"Error preprocessing {self.current_file}: {e}")
            import traceback
            self.log_text.append(f"Traceback: {traceback.format_exc()}")

    def predict_current_file(self):
        if self.preprocessed_spectrum is None:
            self.log_text.append("No preprocessed spectrum to predict. Please preprocess first.")
            return
        try:
            ok = self.ensure_assets_loaded()
            if not ok:
                self.log_text.append("Cannot predict because assets failed to load.")
                return
            model = self._cached_model
            element_map = self._cached_element_map
            target_wavelengths = self._cached_target_wavelengths

            from app.model import predict_with_spatial_info

            self.log_text.append(f"Running prediction with spatial information...")

            # Dapatkan prediksi global DAN spasial (per-wavelength)
            global_pred, spatial_pred = predict_with_spatial_info(model, self.preprocessed_spectrum)

            self.log_text.append(f"Global prediction shape: {global_pred.shape}")
            self.log_text.append(f"Spatial prediction shape: {spatial_pred.shape}")
            self.log_text.append(f"Global pred range: [{global_pred.min():.3f}, {global_pred.max():.3f}]")

            # Format hasil prediksi dengan nama elemen
            element_names = list(element_map.keys())
            result_text = "Predicted Elements (probability > 0.05):\n"
            result_text += "=" * 40 + "\n"
            detected_elements = []

            # Ambil semua elemen dengan threshold rendah
            threshold = 0.05
            for element, probability in zip(element_names, global_pred):
                if probability > threshold:
                    detected_elements.append((element, float(probability)))

            # Sort berdasarkan probability
            detected_elements.sort(key=lambda x: x[1], reverse=True)

            # Jika tidak ada yang terdeteksi, tampilkan top 10
            if not detected_elements:
                sorted_pred = sorted(zip(element_names, global_pred), key=lambda x: x[1], reverse=True)
                result_text = "Top 10 Predicted Elements:\n"
                result_text += "=" * 40 + "\n"
                for element, probability in sorted_pred[:10]:
                    result_text += f"{element:>6s}: {probability:.4f} ({probability*100:.2f}%)\n"
                    detected_elements.append((element, float(probability)))
            else:
                for element, probability in detected_elements:
                    result_text += f"{element:>6s}: {probability:.4f} ({probability*100:.2f}%)\n"

            result_text += "=" * 40 + "\n"
            result_text += f"Total detected: {len(detected_elements)} elements"

            self.results_label.setText(result_text)
            self.results_label.setWordWrap(True)

            # Plot spektrum dengan label elemen MENGGUNAKAN SPATIAL PREDICTION
            self.plot_spectrum_with_labels(
                target_wavelengths,
                self.preprocessed_spectrum,
                detected_elements,
                element_map,
                spatial_pred  # Pass spatial prediction data
            )

            # Update note di tabel
            if self.current_file in self.file_list:
                row = self.file_list.index(self.current_file)
                self.data_table.setItem(row, 2, QtWidgets.QTableWidgetItem("Predicted"))

            self.log_text.append(f"✓ Prediction completed for: {self.current_file}")
            self.log_text.append(f"✓ Found {len(detected_elements)} potential elements")

        except Exception as e:
            self.log_text.append(f"Error during prediction: {e}")
            import traceback
            self.log_text.append(f"Traceback:\n{traceback.format_exc()}")

    def plot_spectrum_with_labels(self, wavelengths, spectrum, detected_elements, element_map, spatial_pred=None):
        """Plot spektrum dengan region selector interaktif, label detail hanya di plot preview"""
        if detected_elements is None or len(detected_elements) == 0:
            self.log_text.append("No elements to label on spectrum")
            return

        try:
            # Clear both plots
            self.plot_widget.clear()
            self.detail_plot_widget.clear()

            # Warna untuk setiap elemen
            colors = [
                (255, 0, 0), (255, 102, 0), (255, 170, 0), (255, 215, 0), (0, 255, 0),
                (0, 255, 255), (0, 136, 255), (0, 0, 255), (136, 0, 255), (255, 0, 255)
            ]

            element_names = list(element_map.keys())

            if spatial_pred is not None:
                self.log_text.append("Setting up interactive region selector...")

                # Store data untuk digunakan saat region berubah
                self.wavelengths = wavelengths
                self.spectrum = spectrum
                self.detected_elements = detected_elements
                self.spatial_pred = spatial_pred
                self.element_names = element_names
                self.colors = colors

                # ========== PLOT UTAMA: Spektrum + Region Selector (TANPA LABEL) ==========
                self.plot_widget.plot(wavelengths, spectrum, pen=pg.mkPen(color=(100, 255, 100), width=2), name='Spectrum')

                # Tambahkan LinearRegionItem untuk memilih region
                # Default region: tengah spektrum ±100nm
                wl_min, wl_max = wavelengths[0], wavelengths[-1]
                wl_center = (wl_min + wl_max) / 2
                # Try to use input values for initial region
                try:
                    input_min = float(self.min_wavelength_input.text())
                    input_max = float(self.max_wavelength_input.text())
                    if input_min < input_max and wl_min <= input_min <= wl_max and wl_min <= input_max <= wl_max:
                        initial_region = (input_min, input_max)
                    else:
                        initial_region = (wl_center - 50, wl_center + 50)
                except ValueError:
                    initial_region = (wl_center - 50, wl_center + 50)

                self.region_selector = pg.LinearRegionItem(
                    values=initial_region,
                    brush=pg.mkBrush(100, 100, 200, 50),  # Semi-transparent blue
                    pen=pg.mkPen(color=(100, 100, 255), width=2)
                )
                self.plot_widget.addItem(self.region_selector)

                # Connect signal untuk update saat region berubah
                self.region_selector.sigRegionChanged.connect(self.update_detail_plot)

                # Setup main plot
                self.plot_widget.setLabel('left', 'Normalized Intensity', **{'font-size': '12pt'})
                self.plot_widget.setLabel('bottom', 'Wavelength (nm)', **{'font-size': '12pt'})
                self.plot_widget.setTitle(
                    'Full Spectrum (Drag the region selector to view details)',
                    **{'font-size': '13pt', 'color': '#333'}
                )
                self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

                # Initial update untuk plot detail
                self.update_detail_plot()

                # Log info
                self.log_text.append(f"✓ Interactive region selector ready")
                self.log_text.append(f"✓ Drag the blue region on main plot to see details")
                self.log_text.append(f"✓ Detail plot will show labels for selected wavelength range")

            else:
                # Fallback
                self.log_text.append("No spatial prediction available")
                self.plot_widget.plot(wavelengths, spectrum, pen='g')
                self.plot_widget.setTitle('Preprocessed Spectrum')

        except Exception as e:
            self.log_text.append(f"Error plotting spectrum: {e}")
            self.log_text.append(traceback.format_exc())

    def update_detail_plot(self):
        """Update plot detail berdasarkan region yang dipilih di plot utama"""
        try:
            # Get peak parameters from inputs
            try:
                detection_threshold = float(self.threshold_input.text())
                min_height = float(self.min_height_input.text())
                min_width = float(self.min_width_input.text())
                max_peaks = int(self.max_peaks_input.text())
                prominence = float(self.prominence_input.text())
                min_distance = float(self.min_distance_input.text())
            except ValueError:
                self.log_text.append("Invalid peak parameters. Using defaults.")
                detection_threshold = 0.4
                min_height = 0.1
                min_width = 1
                max_peaks = 3
                prominence = 0.05
                min_distance = 5

            # Get selected region
            region_range = self.region_selector.getRegion()
            wl_start, wl_end = region_range

            # Find indices dalam range
            mask = (self.wavelengths >= wl_start) & (self.wavelengths <= wl_end)
            indices = np.where(mask)[0]

            if len(indices) == 0:
                return

            # Extract data untuk region ini
            wl_region = self.wavelengths[indices]
            spectrum_region = self.spectrum[indices]
            spatial_pred_region = self.spatial_pred[indices, :]

            # Detect peaks in the spectrum region
            peaks, properties = signal.find_peaks(
                spectrum_region,
                height=min_height,
                width=min_width,
                prominence=prominence,
                distance=min_distance
            )

            # Limit to max_peaks if specified
            if len(peaks) > max_peaks:
                # Sort by prominence and take top max_peaks
                sorted_peaks = sorted(zip(peaks, properties['prominences']), key=lambda x: x[1], reverse=True)[:max_peaks]
                peaks = [p[0] for p in sorted_peaks]

            # Clear detail plot
            self.detail_plot_widget.clear()

            # Plot spektrum di region
            self.detail_plot_widget.plot(
                wl_region, spectrum_region,
                pen=pg.mkPen(color=(100, 255, 100), width=2.5),
                name='Spectrum'
            )

            # Label peaks with detected elements
            all_labels = []
            for peak_idx in peaks:
                wl = wl_region[peak_idx]
                intensity = spectrum_region[peak_idx]

                # Find the element with highest probability around this peak (within ±5 points)
                best_elem = None
                best_prob = 0
                best_color = None
                start_idx = max(0, peak_idx - 5)
                end_idx = min(len(spatial_pred_region), peak_idx + 6)
                for elem, global_prob in self.detected_elements[:10]:
                    elem_idx = self.element_names.index(elem)
                    prob_region = spatial_pred_region[start_idx:end_idx, elem_idx]
                    max_prob = np.max(prob_region)
                    if max_prob > best_prob and max_prob > detection_threshold:
                        best_prob = max_prob
                        best_elem = elem
                        best_color = self.colors[elem_idx % len(self.colors)]

                if best_elem:
                    # Add scatter at peak
                    scatter = pg.ScatterPlotItem(
                        [wl], [intensity],
                        pen=pg.mkPen(color=best_color, width=2),
                        brush=pg.mkBrush(*best_color),
                        size=12,
                        symbol='o'
                    )
                    self.detail_plot_widget.addItem(scatter)

                    # Add label
                    text_label = f"{best_elem}:{best_prob:.1%}\n{wl:.1f}nm"
                    text = pg.TextItem(
                        text=text_label,
                        color=best_color,
                        anchor=(0.5, 1.5),
                        border=pg.mkPen(color=best_color, width=1),
                        fill=pg.mkBrush(0, 0, 0, 150)
                    )
                    text.setPos(wl, intensity)
                    self.detail_plot_widget.addItem(text)

                    all_labels.append(f"@{wl:.1f}nm: {best_elem}")

            # Setup detail plot
            self.detail_plot_widget.setLabel('left', 'Intensity', **{'font-size': '10pt'})
            self.detail_plot_widget.setLabel('bottom', 'Wavelength (nm)', **{'font-size': '10pt'})
            self.detail_plot_widget.setTitle(
                f'Detail View ({wl_start:.1f} - {wl_end:.1f} nm) - {len(all_labels)} peaks labeled',
                **{'font-size': '11pt'}
            )
            self.detail_plot_widget.showGrid(x=True, y=True, alpha=0.3)

            # Auto-range untuk region yang dipilih
            self.detail_plot_widget.setXRange(wl_start, wl_end, padding=0.02)

        except Exception as e:
            self.log_text.append(f"Error updating detail plot: {e}")
            import traceback
            self.log_text.append(f"Traceback:\n{traceback.format_exc()}")

    def update_region_from_inputs(self):
        """Update the region selector based on min and max wavelength inputs"""
        try:
            min_wl = float(self.min_wavelength_input.text())
            max_wl = float(self.max_wavelength_input.text())
            if hasattr(self, 'region_selector'):
                self.region_selector.setRegion((min_wl, max_wl))
        except ValueError:
            pass  # Ignore invalid input

    def export_publication_plot(self):
        """Export the current detail plot (preview) with element labels as a publication-ready scientific figure"""
        try:
            # Check if we have spectrum data and region selector
            if not hasattr(self, 'wavelengths') or not hasattr(self, 'spectrum') or not hasattr(self, 'region_selector'):
                self.log_text.append("No spectrum data or region selector available for export. Please load, process a file, and select a region first.")
                return

            # Get current region
            region_range = self.region_selector.getRegion()
            wl_start, wl_end = region_range

            # Find indices in range
            mask = (self.wavelengths >= wl_start) & (self.wavelengths <= wl_end)
            indices = np.where(mask)[0]

            if len(indices) == 0:
                self.log_text.append("No data in selected region for export.")
                return

            # Extract data for region
            wl_region = self.wavelengths[indices]
            spectrum_region = self.spectrum[indices]
            spatial_pred_region = self.spatial_pred[indices, :]

            # Create matplotlib figure with publication style
            plt.style.use('seaborn-v0_8-paper')  # Use a clean style
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

            # Plot the spectrum region
            ax.plot(wl_region, spectrum_region, color='black', linewidth=1.0)

            # Get detection threshold and peak parameters
            try:
                detection_threshold = float(self.threshold_input.text())
                min_height = float(self.min_height_input.text())
                min_width = float(self.min_width_input.text())
                max_peaks = int(self.max_peaks_input.text())
                prominence = float(self.prominence_input.text())
                min_distance = float(self.min_distance_input.text())
            except ValueError:
                detection_threshold = 0.4
                min_height = 0.1
                min_width = 1
                max_peaks = 3
                prominence = 0.05
                min_distance = 5

            # Detect peaks in the spectrum region
            peaks, properties = signal.find_peaks(
                spectrum_region,
                height=min_height,
                width=min_width,
                prominence=prominence,
                distance=min_distance
            )

            # Limit to max_peaks
            if len(peaks) > max_peaks:
                sorted_peaks = sorted(zip(peaks, properties['prominences']), key=lambda x: x[1], reverse=True)[:max_peaks]
                peaks = [p[0] for p in sorted_peaks]

            # Add detected elements annotations at peaks
            colors = [
                (255, 0, 0), (255, 102, 0), (255, 170, 0), (255, 215, 0), (0, 255, 0),
                (0, 255, 255), (0, 136, 255), (0, 0, 255), (136, 0, 255), (255, 0, 255)
            ]

            for peak_idx in peaks:
                wl = wl_region[peak_idx]
                intensity = spectrum_region[peak_idx]

                # Find the element with highest probability around this peak
                best_elem = None
                best_prob = 0
                best_color = None
                start_idx = max(0, peak_idx - 5)
                end_idx = min(len(spatial_pred_region), peak_idx + 6)
                for elem, global_prob in self.detected_elements[:10]:
                    elem_idx = self.element_names.index(elem)
                    prob_region = spatial_pred_region[start_idx:end_idx, elem_idx]
                    max_prob = np.max(prob_region)
                    if max_prob > best_prob and max_prob > detection_threshold:
                        best_prob = max_prob
                        best_elem = elem
                        best_color = colors[elem_idx % len(colors)]

                if best_elem:
                    ax.scatter(wl, intensity, color=np.array(best_color)/255, s=60, zorder=5, edgecolors='black', linewidth=0.5)

                    # Create annotation text
                    text_label = f"{best_elem}:{best_prob:.1%}\n{wl:.1f}nm"

                    # Position annotation above the point
                    ax.annotate(text_label,
                              xy=(wl, intensity),
                              xytext=(5, 20),
                              textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'),
                              fontsize=9, ha='left', va='bottom',
                              arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

            # Customize plot for publication
            ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Normalized Intensity', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_xlim(wl_start, wl_end)

            # Add file info
            if self.current_file:
                ax.text(0.02, 0.98, f'File: {self.current_file}\nRegion: {wl_start:.1f} - {wl_end:.1f} nm',
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

            plt.tight_layout()

            # Save dialog
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Publication Plot", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)")
            if file_path:
                import os
                if os.path.isdir(file_path):
                    self.log_text.append("Selected path is a directory. Please select a valid file name.")
                    plt.close(fig)
                    return
                # Determine format from extension
                if file_path.lower().endswith('.pdf'):
                    plt.savefig(file_path, format='pdf', bbox_inches='tight', dpi=300)
                elif file_path.lower().endswith('.svg'):
                    plt.savefig(file_path, format='svg', bbox_inches='tight')
                else:
                    plt.savefig(file_path, format='png', bbox_inches='tight', dpi=300)

                self.log_text.append(f"✓ Publication plot exported to: {file_path}")
            else:
                self.log_text.append("Export cancelled.")

            plt.close(fig)  # Close the figure to free memory

        except Exception as e:
            self.log_text.append(f"Error exporting publication plot: {e}")
            import traceback
            self.log_text.append(f"Traceback:\n{traceback.format_exc()}")
