from __future__ import annotations
from typing import TYPE_CHECKING, Any
import os

import numpy as np
import pyqtgraph as pg

# Import untuk ekspor plot publikasi
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    
    # Set matplotlib untuk publikasi
    matplotlib.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'lines.linewidth': 2.0,
        'patch.linewidth': 1.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.transparent': False
    })
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

if TYPE_CHECKING:
    from PySide6.QtCore import Slot, QSize, Signal, Qt, QTimer
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGroupBox,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QPushButton,
        QCheckBox,
        QSizePolicy,
        QFileDialog,
        QMessageBox,
        QScrollArea,
        QLabel,
    )
else:
    from PySide6 import QtCore as _QtCore  # type: ignore
    from PySide6 import QtWidgets as _QtWidgets  # type: ignore
    from PySide6 import QtGui as _QtGui  # type: ignore

    Slot = _QtCore.Slot
    Signal = _QtCore.Signal
    QSize = _QtCore.QSize
    Qt = _QtCore.Qt
    QTimer = _QtCore.QTimer

    QColor = _QtGui.QColor

    QWidget = _QtWidgets.QWidget
    QVBoxLayout = _QtWidgets.QVBoxLayout
    QHBoxLayout = _QtWidgets.QHBoxLayout
    QLabel = _QtWidgets.QLabel
    QSizePolicy = _QtWidgets.QSizePolicy
    QScrollArea = _QtWidgets.QScrollArea
    QGroupBox = _QtWidgets.QGroupBox
    QTableWidget = _QtWidgets.QTableWidget
    QTableWidgetItem = _QtWidgets.QTableWidgetItem
    QHeaderView = _QtWidgets.QHeaderView
    QPushButton = _QtWidgets.QPushButton
    QCheckBox = _QtWidgets.QCheckBox
    QFileDialog = _QtWidgets.QFileDialog
    QMessageBox = _QtWidgets.QMessageBox
    QSizePolicy = _QtWidgets.QSizePolicy


class KeyNavigablePlotWidget(pg.PlotWidget):
    """Plot widget dengan kemampuan navigasi menggunakan keyboard"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Izinkan widget menerima keyboard focus
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.roi_parent = None

    def keyPressEvent(self, ev):
        """Handle key press events untuk navigasi ROI"""
        # Pastikan parent memiliki ROI dan fungsi shift ROI
        if self.roi_parent is not None and hasattr(self.roi_parent, 'shift_roi'):
            key = ev.key()
            if key == Qt.Key.Key_Left:
                self.roi_parent.shift_roi(-1)  # Geser ROI ke kiri
                ev.accept()
                return
            elif key == Qt.Key.Key_Right:
                self.roi_parent.shift_roi(1)   # Geser ROI ke kanan
                ev.accept()
                return
            # Tambah/kurangi lebar step pergeseran
            elif key == Qt.Key.Key_Up:
                if hasattr(self.roi_parent, '_roi_shift_step'):
                    self.roi_parent._roi_shift_step *= 1.5
                    print(f"ROI shift step increased to: {self.roi_parent._roi_shift_step:.2f} nm")
                    
                    # Emit signal to update UI
                    if hasattr(self.roi_parent, 'sensitivityChanged'):
                        self.roi_parent.sensitivityChanged.emit(self.roi_parent._roi_shift_step)
                        
                    # Update status label if available
                    if hasattr(self.roi_parent, '_roi_status_label') and self.roi_parent._roi_status_label:
                        self.roi_parent._roi_status_label.setText(f"{self.roi_parent._roi_shift_step:.1f} nm")
                        self.roi_parent._roi_status_label.setStyleSheet("font-weight: bold; color: white; background-color: #009900; padding: 3px 8px; border-radius: 3px;")  # Green when increased
                        # Kembalikan warna setelah beberapa detik
                        QTimer.singleShot(1500, lambda: self.roi_parent._roi_status_label.setStyleSheet("font-weight: bold; color: white; background-color: #0066cc; padding: 3px 8px; border-radius: 3px;"))
                ev.accept()
                return
            elif key == Qt.Key.Key_Down:
                if hasattr(self.roi_parent, '_roi_shift_step'):
                    self.roi_parent._roi_shift_step /= 1.5
                    print(f"ROI shift step decreased to: {self.roi_parent._roi_shift_step:.2f} nm")
                    
                    # Emit signal to update UI
                    if hasattr(self.roi_parent, 'sensitivityChanged'):
                        self.roi_parent.sensitivityChanged.emit(self.roi_parent._roi_shift_step)
                        
                    # Update status label if available
                    if hasattr(self.roi_parent, '_roi_status_label') and self.roi_parent._roi_status_label:
                        self.roi_parent._roi_status_label.setText(f"{self.roi_parent._roi_shift_step:.1f} nm")
                        self.roi_parent._roi_status_label.setStyleSheet("font-weight: bold; color: white; background-color: #CC0000; padding: 3px 8px; border-radius: 3px;")  # Red when decreased
                        # Kembalikan warna setelah beberapa detik
                        QTimer.singleShot(1500, lambda: self.roi_parent._roi_status_label.setStyleSheet("font-weight: bold; color: white; background-color: #0066cc; padding: 3px 8px; border-radius: 3px;"))
                ev.accept()
                return
        # Untuk key lainnya, biarkan plot widget menanganinya
        super().keyPressEvent(ev)

class SquarePlotWidget(pg.PlotWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sp = self.sizePolicy()
        sp.setHeightForWidth(True)
        sp.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        sp.setVerticalPolicy(QSizePolicy.Policy.Expanding)
        self.setSizePolicy(sp)

    def hasHeightForWidth(self) -> bool:  # type: ignore[override]
        return True

    def heightForWidth(self, w: int) -> int:  # type: ignore[override]
        return int(w)

    def minimumSizeHint(self):  # type: ignore[override]
        s = super().minimumSizeHint()
        side = min(s.width(), s.height())
        return QSize(side, side)

    def sizeHint(self):  # type: ignore[override]
        return QSize(360, 360)


class ResultsPanel(QWidget):
    # Add signals to notify when region changes for zoom plot and sensitivity changes
    regionChanged = Signal(float, float)
    sensitivityChanged = Signal(float)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # (wmin, wmax, ymin, ymax)
        self._data_bounds = None
        # Guard to avoid recursive ROI updates
        self._roi_updating = False
        # Track whether we've applied the default ROI for the current data
        self._roi_initialized = False
        # Initialize the roi attribute
        self.roi = None
        # Signal proxy untuk rate-limiting
        self._region_proxy = None
        # Store data arrays for zoom plot
        self._data_arrays = None
        # Store peaks for zoom plot
        self._peaks = None
        # Simpan posisi zoom terakhir untuk dipertahankan saat parameter diubah
        self._last_zoom_region = (200.0, 900.0)  # Default 200-900 nm
        # Geser ROI dengan keyboard (dalam nm)
        self._roi_shift_step = 5.0  # Langkah pergeseran default (nm)
        
        # Status label untuk menampilkan sensitivitas ROI
        self._roi_status_label = None
        
        # Set size policy to allow proper scrolling
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        
        self._build_ui()

    def _build_ui(self):
        # Single layout for the results panel
        layout = QVBoxLayout(self)

        # Create a single plot widget dengan keyboard navigation
        self.plot_widget = KeyNavigablePlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setTitle("Plot Utama")
        
        # Status area untuk ROI dan sensitivitas
        status_layout = QHBoxLayout()
        
        # Label instruksi (kosong)
        instructions_label = QLabel("")
        
        # Buat label untuk nilai sensitivitas dengan gaya yang lebih menonjol
        sensitivity_title = QLabel("SENSITIVITAS:")
        sensitivity_title.setStyleSheet("font-weight: bold;")
        
        # Buat label nilai dengan highlight khusus
        self._roi_status_label = QLabel(f"{self._roi_shift_step:.1f} nm")
        self._roi_status_label.setStyleSheet("font-weight: bold; color: white; background-color: #0066cc; padding: 3px 8px; border-radius: 3px;")
        
        # Gabungkan dalam layout horizontal
        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(sensitivity_title)
        sensitivity_layout.addWidget(self._roi_status_label)
        sensitivity_layout.addStretch(1)
        
        # Tambahkan widget ke layout status
        status_layout.addWidget(instructions_label)
        status_layout.addLayout(sensitivity_layout)
        
        # Tambahkan tombol ekspor publikasi - sudah ada di bagian atas
        
        status_layout.addStretch(1)
        
        layout.addLayout(status_layout)
        
        # Configure plot to fill the available space
        self.plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Set default zoom range for X-axis (200–900 nm)
        self.plot_widget.setXRange(200, 900)
        # Allow dynamic Y-axis scaling
        self.plot_widget.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        
        # Hubungkan plot dengan ResultsPanel
        self.plot_widget.roi_parent = self
        
        layout.addWidget(self.plot_widget)
        
        # Create results table for displaying prediction/validation results
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #f0f8ff;
                gridline-color: #d0d0d0;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: #e6f3ff;
                padding: 6px;
                border: none;
                font-weight: bold;
            }
        """)
        
        # Initially hide the table
        self.results_table.setVisible(False)
        
        # Add table to layout with smaller height
        self.results_table.setMaximumHeight(200)
        layout.addWidget(self.results_table)
        
        # Only set the layout once
        self.setLayout(layout)

    def on_region_changed(self):
        if self.roi is None:
            return
        try:
            # Untuk LinearRegionItem, mendapatkan region lebih sederhana
            region_vals = self.roi.getRegion()
            # Abaikan type checking untuk baris ini
            x0, x1 = float(region_vals[0]), float(region_vals[1])  # type: ignore
            
            # Pastikan x0 < x1
            if x0 > x1:
                x0, x1 = x1, x0
                
            # Emit signal to update zoom plot in MainWindow
            self.regionChanged.emit(x0, x1)
            # Debug message to verify signal emission
            print(f"Region changed: x0={x0}, x1={x1}")

            # Clamp to data bounds so the zoom view never goes out of range
            eps = 1e-9
            if self._data_bounds is not None:
                wmin, wmax, ymin, ymax = self._data_bounds
                minw = max(eps, 0.005 * max(eps, wmax - wmin))  # Sedikit lebih besar untuk mengurangi sensitivitas
                
                # Pastikan rentang minimum
                if (x1 - x0) < minw:
                    x1 = x0 + minw
                
                # Batasi dalam rentang data
                x0 = max(wmin, min(x0, wmax - minw))
                x1 = min(wmax, max(x1, wmin + minw))
                
                # Jika perlu memperbaiki posisi ROI
                try:
                    if not self._roi_updating:
                        self._roi_updating = True
                        try:
                            self.roi.blockSignals(True)
                            # Gunakan setRegion untuk LinearRegionItem
                            self.roi.setRegion((x0, x1))
                        finally:
                            self.roi.blockSignals(False)
                            self._roi_updating = False
                except Exception:
                    self._roi_updating = False

            # Simpan region untuk zoom plot
            self._last_zoom_region = (x0, x1)
            
            # Update status label jika tersedia - hanya tampilkan nilai sensitivitas
            if hasattr(self, '_roi_status_label') and self._roi_status_label is not None:
                self._roi_status_label.setText(f"{self._roi_shift_step:.1f} nm")
            
            # Perbarui tampilan plot utama agar bergeser mengikuti ROI jika diperlukan
            vbz = self.plot_widget.getViewBox()
            current_view_range = vbz.viewRange()
            current_view_width = current_view_range[0][1] - current_view_range[0][0]
            
            # Jika ROI keluar dari area tampilan atau terlalu dekat dengan tepi,
            # geser tampilan untuk mengikuti ROI
            margin = current_view_width * 0.2  # Margin 20% dari lebar tampilan
            need_adjustment = (x0 < current_view_range[0][0] + margin or
                             x1 > current_view_range[0][1] - margin)
            
            if need_adjustment:
                center = (x0 + x1) / 2
                # Pertahankan lebar tampilan yang ada
                half_width = current_view_width / 2
                vbz.setXRange(center - half_width, center + half_width, padding=0)
                
            # Untuk sumbu Y, kita biarkan auto-range agar semua data dalam rentang X terlihat
            vbz.enableAutoRange(axis='y', enable=True)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return

    @Slot(dict)
    def update_preview(self, results: dict[str, Any]):
        wl = results.get("wavelengths")
        y = results.get("spectrum_data")

        self.plot_widget.clear()

        if wl is None or y is None:
            return

        try:
            self.plot_widget.plot(wl, y, pen=pg.mkPen('b', width=1.5))
        except (RuntimeError, TypeError, ValueError):
            pass

        try:
            wl_arr = np.asarray(wl, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            if wl_arr.size >= 2:
                # Cek apakah ini dataset yang benar-benar baru (berbeda panjang data)
                is_new_dataset = (not hasattr(self, '_data_arrays') or 
                                 self._data_arrays is None or 
                                 len(self._data_arrays[0]) != len(wl_arr))
                
                print(f"Processing data: new dataset = {is_new_dataset}, data length = {len(wl_arr)}")
                
                # Hanya reset flag ROI jika ini benar-benar dataset baru
                if is_new_dataset:
                    self._roi_initialized = False
                
                try:
                    self._zoom_curve = self.plot_widget.plot(wl_arr, y_arr, pen=pg.mkPen('b', width=1.5))
                except (RuntimeError, TypeError, ValueError):
                    pass

                wmin, wmax = float(np.nanmin(wl_arr)), float(np.nanmax(wl_arr))
                ymin, ymax = float(np.nanmin(y_arr)), float(np.nanmax(y_arr))
                self._data_bounds = (wmin, wmax, ymin, ymax)
                # Store data arrays for zoom plot
                self._data_arrays = [wl_arr, y_arr]

                # Prepare ROI if needed and set its default to 200-900 nm (clamped to data)
                if self.roi is None:
                    try:
                        self._create_default_roi(wl_arr, y_arr, wmin, wmax, ymin, ymax)
                        if self.roi is not None:
                            # Gunakan SignalProxy untuk membatasi kecepatan update dan mengurangi sensitivitas
                            self._region_proxy = pg.SignalProxy(
                                self.roi.sigRegionChanged,
                                rateLimit=60,  # Batasi ke 60 fps maksimum
                                slot=self.on_region_changed
                            )
                    except (RuntimeError, TypeError, ValueError):
                        self.roi = None
                # Apply default ROI only untuk dataset yang benar-benar baru
                if not self._roi_initialized and self.roi is not None:
                    try:
                        # Set ROI default untuk dataset baru
                        self._set_default_roi(wl_arr, y_arr, wmin, wmax, ymin, ymax)
                        self._roi_initialized = True
                    except Exception:
                        pass
                elif self.roi is not None:
                    try:
                        # Untuk dataset yang sama dengan parameter berbeda, 
                        # pertahankan posisi ROI saat ini
                        current_region = self.roi.getRegion()
                        print(f"Preserving ROI position: {current_region}")
                        # Batasi ROI dalam rentang data baru jika diperlukan
                        x0 = max(wmin, min(current_region[0], wmax))
                        x1 = max(wmin, min(current_region[1], wmax))
                        # Update region tanpa mengaktifkan signal
                        self.roi.blockSignals(True)
                        self.roi.setRegion((x0, x1))
                        self.roi.blockSignals(False)
                        # Pastikan posisi region dipertahankan untuk digunakan pada zoom plot
                        self._last_zoom_region = (x0, x1)
                    except Exception as e:
                        print(f"Error preserving ROI: {e}")
                        pass

                if self.roi is not None:
                    try:
                        if self.roi.scene() is None:
                            self.plot_widget.addItem(self.roi)
                    except (RuntimeError, AttributeError):
                        pass

                # Panggil on_region_changed untuk update zoom plot dengan region saat ini
                self.on_region_changed()
                
                # Perbarui status label untuk menampilkan nilai sensitivitas setelah pertama kali load
                if hasattr(self, '_roi_status_label') and self._roi_status_label is not None:
                    try:
                        region = self.roi.getRegion()
                        x0, x1 = float(region[0]), float(region[1])
                        roi_width = x1 - x0
                        self._roi_status_label.setText(f"ROI: {x0:.1f}-{x1:.1f} nm (lebar: {roi_width:.1f} nm) | Sensitivitas: {self._roi_shift_step:.1f} nm")
                    except Exception:
                        pass
                
                # Tambahkan timer untuk memastikan zoom plot terupdate setelah semua rendering selesai
                timer = pg.QtCore.QTimer()
                timer.setSingleShot(True)
                timer.timeout.connect(self.on_region_changed)
                timer.start(100)  # 100ms delay untuk memastikan UI telah terupdate
                
                # Jika region sudah ada, pastikan zoom plot diperbarui
                if not is_new_dataset and self.roi is not None:
                    try:
                        # Panggil secara langsung untuk memastikan zoom plot terupdate
                        region = self.roi.getRegion()
                        print(f"Forcing zoom update after parameter change: {region}")
                        self.regionChanged.emit(float(region[0]), float(region[1]))
                    except Exception as e:
                        print(f"Error updating zoom after parameters change: {e}")
                        pass
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass

    def _preferred_xrange(self, wmin: float, wmax: float) -> tuple[float, float]:
        # Use last zoom region if available, otherwise default to 200-900 nm
        if hasattr(self, '_last_zoom_region') and self._last_zoom_region:
            # Clamp last region to new data bounds
            x0 = max(wmin, min(self._last_zoom_region[0], wmax))
            x1 = max(wmin, min(self._last_zoom_region[1], wmax))
            # Check if region is valid
            if x1 - x0 > 1e-6 and x0 >= wmin and x1 <= wmax:
                print(f"Using last zoom region: {x0:.2f}-{x1:.2f}")
                return x0, x1
        
        # Default preference 200-900 nm, clamped to data bounds
        x0 = max(wmin, 200.0)
        x1 = min(wmax, 900.0)
        if x1 - x0 <= 1e-6:
            # Fallback to middle 60% if insufficient overlap
            span = max(1e-6, wmax - wmin)
            x0 = wmin + 0.2 * span
            x1 = wmin + 0.8 * span
        return x0, x1

    def _create_default_roi(self, wl_arr: np.ndarray, y_arr: np.ndarray, wmin: float, wmax: float, ymin: float, ymax: float):
        dx = max(1e-9, wmax - wmin)
        dy = max(1e-9, ymax - ymin)
        x0, x1 = self._preferred_xrange(wmin, wmax)
        mask = (wl_arr >= x0) & (wl_arr <= x1)
        if np.any(mask):
            ymin_s = float(np.nanmin(y_arr[mask])); ymax_s = float(np.nanmax(y_arr[mask]))
            if np.isfinite(ymin_s) and np.isfinite(ymax_s) and ymax_s > ymin_s:
                ymin, ymax = ymin_s, ymax_s
                dy = max(1e-9, ymax - ymin)
        # Menggunakan LinearRegionItem yang lebih stabil untuk ROI
        # Ini hanya akan memungkinkan pemilihan rentang panjang gelombang (sumbu X)
        self.roi = pg.LinearRegionItem(
            values=(x0, x1),
            orientation="vertical", 
            brush=pg.mkBrush(color=(255, 255, 0, 50)),  # Warna kuning transparan
            pen=pg.mkPen(color=(255, 165, 0), width=2.5),  # Border oranye tebal
            hoverBrush=pg.mkBrush(color=(255, 255, 0, 70)),  # Warna saat hover
            movable=True,
            bounds=(wmin, wmax)  # Batasi dalam rentang data
        )
        # Kurangi sensitivitas dengan menambahkan ratelimit pada sinyal
        self.roi.setMouseHover(True)  # Aktifkan efek hover

    def _set_default_roi(self, wl_arr: np.ndarray, y_arr: np.ndarray, wmin: float, wmax: float, ymin: float, ymax: float):
        if self.roi is None:
            return
        dx = max(1e-9, wmax - wmin)
        dy = max(1e-9, ymax - ymin)
        x0, x1 = self._preferred_xrange(wmin, wmax)
        mask = (wl_arr >= x0) & (wl_arr <= x1)
        if np.any(mask):
            ymin_s = float(np.nanmin(y_arr[mask])); ymax_s = float(np.nanmax(y_arr[mask]))
            if np.isfinite(ymin_s) and np.isfinite(ymax_s) and ymax_s > ymin_s:
                ymin, ymax = ymin_s, ymax_s
                dy = max(1e-9, ymax - ymin)
        try:
            self.roi.blockSignals(True)
            # Untuk LinearRegionItem, kita hanya perlu menetapkan region
            self.roi.setRegion((x0, x1))
        finally:
            self.roi.blockSignals(False)
        # Sync zoom to ROI
        self.on_region_changed()

    @Slot(dict)
    def update_full_results(self, results: dict[str, Any]):
        # First update the preview with spectrum and peaks
        self.update_preview(results)
        
        # Store downsampled data if available (used for accurate prediction in zoom)
        if "downsampled_wavelengths" in results and "downsampled_intensities" in results:
            self._downsampled_data = [
                results["downsampled_wavelengths"],
                results["downsampled_intensities"]
            ]
            print(f"Stored downsampled data: {len(self._downsampled_data[0])} points")
        else:
            self._downsampled_data = None
            print("No downsampled data available, using original resolution")

        # Store data and annotations for future export
        # Enable export button when data is available
        # Will be handled by main window export button

        # Handle prediction/validation results
        table_data = (
            results.get("validation_table") if results.get("validation_table") else results.get("prediction_table", [])
        )
        
        # Display annotations on the plot
        annotations = results.get("annotations", [])
        # Always call _add_prediction_annotations to handle clearing old annotations
        self._add_prediction_annotations(annotations)
        
        # Show summary information
        analysis_mode = results.get("analysis_mode", "predict")
        summary_metrics = results.get("summary_metrics", {})
        
        # Update results table with prediction/validation data
        self._update_results_table(table_data, analysis_mode, summary_metrics)
        
        # Update plot title with prediction results
        if table_data:
            try:
                # Count unique elements
                unique_elements = set()
                for row in table_data:
                    element = row.get("Elemen", "")
                    if element and element != "-":
                        unique_elements.add(element)
                
                element_count = len(unique_elements)
                element_list = ", ".join(sorted(unique_elements)[:5])  # Show first 5 elements
                if len(unique_elements) > 5:
                    element_list += f" (+{len(unique_elements)-5} lainnya)"
                
                if analysis_mode == "validate" and summary_metrics:
                    # Show validation metrics
                    precision = summary_metrics.get("Precision", "N/A")
                    recall = summary_metrics.get("Recall", "N/A")
                    f1 = summary_metrics.get("F1-Score", "N/A")
                    title = f"VALIDASI: {element_count} elemen | P={precision} R={recall} F1={f1}"
                else:
                    # Show prediction results
                    title = f"PREDIKSI: {element_count} elemen ditemukan"
                    if element_list:
                        title += f" - {element_list}"
                
                self.plot_widget.setTitle(title)
                
                # Print results to console for debugging
                print(f"\n=== HASIL {analysis_mode.upper()} ===")
                print(f"Elemen ditemukan: {element_count}")
                if unique_elements:
                    print(f"Daftar elemen: {', '.join(sorted(unique_elements))}")
                    
                if summary_metrics:
                    print("Metrik validasi:")
                    for key, value in summary_metrics.items():
                        print(f"  {key}: {value}")
                        
                print("Detail tabel:")
                for i, row in enumerate(table_data[:10]):  # Show first 10 rows
                    print(f"  {i+1}. {row}")
                if len(table_data) > 10:
                    print(f"  ... dan {len(table_data)-10} baris lainnya")
                print("=== SELESAI ===\n")
                    
            except (TypeError, ValueError, AttributeError, IndexError) as e:
                print(f"Error processing prediction results: {e}")
                self.plot_widget.setTitle("PREDIKSI: Error memproses hasil")
        else:
            self.plot_widget.setTitle("PREDIKSI: Tidak ada elemen terdeteksi")
            print("\n=== HASIL PREDIKSI ===")
            print("Tidak ada elemen terdeteksi dengan threshold yang ditetapkan")
            print("=== SELESAI ===\n")
            # Hide table when no results
            self.results_table.setVisible(False)

    def _update_results_table(self, table_data: list[dict], analysis_mode: str, summary_metrics: dict):
        """Update the results table with prediction/validation data"""
        if not table_data:
            self.results_table.setVisible(False)
            return
        
        # Show the table
        self.results_table.setVisible(True)
        
        # Get column headers from first row
        if table_data:
            headers = list(table_data[0].keys())
            self.results_table.setColumnCount(len(headers))
            self.results_table.setHorizontalHeaderLabels(headers)
            
            # Set row count
            self.results_table.setRowCount(len(table_data))
            
            # Fill table with data
            for row_idx, row_data in enumerate(table_data):
                for col_idx, header in enumerate(headers):
                    value = row_data.get(header, "")
                    # Format value based on type
                    if isinstance(value, float):
                        if header in ["Wavelength (nm)", "Panjang Gelombang (nm)"]:
                            formatted_value = f"{value:.2f}"
                        elif header in ["Confidence", "Kepercayaan"]:
                            formatted_value = f"{value:.3f}"
                        else:
                            formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    
                    item = QTableWidgetItem(formatted_value)
                    
                    # Color code based on confidence or element type
                    if header == "Elemen" and value and value != "-":
                        item.setBackground(QColor(200, 255, 200))  # Light green for detected elements
                    elif header in ["Confidence", "Kepercayaan"] and isinstance(value, float):
                        if value > 0.8:
                            item.setBackground(QColor(144, 238, 144))  # Light green for high confidence
                        elif value > 0.6:
                            item.setBackground(QColor(255, 255, 144))  # Light yellow for medium confidence
                        else:
                            item.setBackground(QColor(255, 182, 193))  # Light red for low confidence
                    
                    self.results_table.setItem(row_idx, col_idx, item)
            
            # Resize columns to fit content
            self.results_table.resizeColumnsToContents()
            
            # Add summary metrics as additional info if available
            if summary_metrics and analysis_mode == "validate":
                # Add a separator row
                current_rows = self.results_table.rowCount()
                self.results_table.setRowCount(current_rows + 2)
                
                # Add summary header
                summary_item = QTableWidgetItem("=== RINGKASAN VALIDASI ===")
                summary_item.setBackground(QColor(173, 216, 230))  # Light blue
                summary_item.setForeground(QColor(0, 0, 0))
                self.results_table.setItem(current_rows, 0, summary_item)
                self.results_table.setSpan(current_rows, 0, 1, len(headers))
                
                # Add metrics
                metrics_text = " | ".join([f"{k}: {v}" for k, v in summary_metrics.items()])
                metrics_item = QTableWidgetItem(metrics_text)
                metrics_item.setBackground(QColor(230, 230, 250))  # Lavender
                self.results_table.setItem(current_rows + 1, 0, metrics_item)
                self.results_table.setSpan(current_rows + 1, 0, 1, len(headers))

    def _add_prediction_annotations(self, annotations):
        """Add prediction annotations to the zoom plot only (not main plot)"""
        try:
            # Store annotations for use in zoom plot
            self._stored_annotations = annotations
            print(f"Stored {len(annotations)} annotations for zoom plot")
            
            # Clear old annotations if no new ones
            if not annotations:
                self._stored_annotations = []
                print("Cleared old annotations - no new predictions")
            
            # Don't add annotations to main plot anymore
            # Annotations will be added to zoom plot in update_zoom_plot method
            
        except Exception as e:
            print(f"Error storing annotations: {e}")

    def on_zoom_toggle(self, checked: bool):
        try:
            if checked and self.roi is not None:
                if self.roi.scene() is None:
                    self.plot_widget.addItem(self.roi)
            else:
                if self.roi is not None and self.roi.scene() is not None:
                    self.plot_widget.removeItem(self.roi)
        except (RuntimeError, AttributeError):
            pass

    def on_reset_clicked(self):
        try:
            vb = self.plot_widget.getViewBox()
            vb.autoRange()
        except (RuntimeError, AttributeError):
            pass
        try:
            if self.roi is not None and self._data_bounds is not None:
                wmin, wmax, ymin, ymax = self._data_bounds
                # Reset ROI to default preferred range
                self._set_default_roi(np.array([wmin, wmax]), np.array([ymin, ymax]), wmin, wmax, ymin, ymax)
                self._roi_initialized = True
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass

    def set_show_baseline_overlay(self, _on: bool):
        # Placeholder kept for compatibility; baseline overlay not drawn in this version
        return
        
    def shift_roi(self, direction):
        """Geser ROI ke kiri atau kanan sesuai direction (-1=kiri, 1=kanan)"""
        if self.roi is None or not hasattr(self, '_roi_shift_step'):
            return
            
        try:
            # Ambil region saat ini
            region = self.roi.getRegion()
            x0, x1 = float(region[0]), float(region[1])
            width = x1 - x0
            
            # Hitung pergeseran
            shift_amount = direction * self._roi_shift_step
            
            # Jika ada data bounds, pastikan tidak keluar dari batas data
            if self._data_bounds is not None:
                wmin, wmax, _, _ = self._data_bounds
                
                new_x0 = max(wmin, region[0] + shift_amount)
                new_x1 = min(wmax, region[1] + shift_amount)
                
                # Jika akan keluar dari batas, jangan geser
                if new_x0 <= wmin and direction < 0:
                    print("Sudah mencapai batas kiri data")
                    return
                if new_x1 >= wmax and direction > 0:
                    print("Sudah mencapai batas kanan data")
                    return
                    
                # Set region baru
                self.roi.blockSignals(True)
                self.roi.setRegion((new_x0, new_x1))
                self.roi.blockSignals(False)
                
                # Update zoom plot dan simpan posisi baru
                self._last_zoom_region = (new_x0, new_x1)
                self.regionChanged.emit(new_x0, new_x1)
                
                # Update visual feedback
                # Tambahkan highlight sementara pada ROI saat digerakkan
                try:
                    original_brush = self.roi.brush
                    self.roi.setBrush(pg.mkBrush(color=(255, 100, 0, 80)))  # Lebih mencolok
                    # Kembalikan warna setelah sebentar
                    QTimer.singleShot(300, lambda roi=self.roi, brush=original_brush: roi.setBrush(brush) if roi and hasattr(roi, 'setBrush') else None)
                except Exception:
                    pass  # Abaikan error jika terjadi masalah dengan brush
                
                # Tidak perlu mengubah label status di sini karena hanya menampilkan nilai sensitivitas
                
                # Update tampilan plot utama untuk mengikuti pergeseran ROI
                vb = self.plot_widget.getViewBox()
                current_view_range = vb.viewRange()
                current_view_width = current_view_range[0][1] - current_view_range[0][0]
                
                # Hanya geser tampilan jika ROI berada di luar area pandang atau terlalu dekat dengan tepi
                margin = current_view_width * 0.2  # Margin 20% dari lebar tampilan
                if new_x0 < current_view_range[0][0] + margin or new_x1 > current_view_range[0][1] - margin:
                    # Geser tampilan untuk mengikuti ROI
                    center = (new_x0 + new_x1) / 2
                    half_width = current_view_width / 2
                    vb.setXRange(center - half_width, center + half_width, padding=0)
                
                print(f"ROI digeser {direction > 0 and 'kanan' or 'kiri'}: {abs(shift_amount):.2f} nm")
        except Exception as e:
            print(f"Error shifting ROI: {e}")
            
    def update_zoom_plot(self, zoom_widget, x0, x1):
        """Update the zoom preview plot with data from the selected region"""
        print(f"update_zoom_plot called: x0={x0}, x1={x1}, has data: {hasattr(self, '_data_arrays')}")
        if not hasattr(self, '_data_arrays') or not self._data_arrays:
            print("No data arrays available")
            return
            
        # Simpan posisi region terakhir untuk dipertahankan saat parameter diubah
        old_region = getattr(self, '_last_zoom_region', (200.0, 900.0))
        self._last_zoom_region = (x0, x1)
        print(f"Zoom region changed: from {old_region} to {self._last_zoom_region}")
        
        # Check if we have downsampled data for prediction accuracy
        if hasattr(self, '_downsampled_data') and self._downsampled_data:
            print("Using downsampled data for zoom plot (better for prediction)")
            wl = np.asarray(self._downsampled_data[0])
            intens = np.asarray(self._downsampled_data[1])
        else:
            print("Using original resolution data for zoom plot")
            wl = np.asarray(self._data_arrays[0])
            intens = np.asarray(self._data_arrays[1])
        
        if len(wl) == 0 or len(intens) == 0:
            return
            
        # Ensure x0 < x1
        if x0 > x1:
            x0, x1 = x1, x0
            
        # Create mask for the data in the selected region
        mask = (wl >= x0) & (wl <= x1)
        
        if not np.any(mask):
            zoom_widget.setTitle(f"Tidak ada data di {x0:.2f}-{x1:.2f} nm")
            return
            
        # Clear previous plot
        zoom_widget.clear()
        
        # Plot the selected region with extra prominent appearance
        data_source = "downsampled" if hasattr(self, '_downsampled_data') and self._downsampled_data else "original"
        zoom_widget.setTitle(f"ZOOM: {x0:.2f} - {x1:.2f} nm ({data_source})")
        print(f"Plotting data in zoom: {np.sum(mask)} points in range ({data_source} resolution)")
        
        # Use a very bright color with thicker line
        bright_pen = pg.mkPen(color=(255, 0, 0), width=3)
        plot_item = zoom_widget.plot(wl[mask], intens[mask], pen=bright_pen)
        print(f"Zoom plot created with data: {plot_item is not None}")
        
        # Add peak points if available
        if hasattr(self, '_peaks') and self._peaks:
            pwl = np.asarray(self._peaks[0])
            pint = np.asarray(self._peaks[1])
            pmask = (pwl >= x0) & (pwl <= x1)
            if np.any(pmask):
                zoom_widget.plot(
                    pwl[pmask],
                    pint[pmask],
                    pen=None,
                    symbol="o",
                    symbolBrush="r",
                    symbolPen="r",
                    symbolSize=6,
                )
        
        # Add prediction annotations ONLY to zoom plot
        if hasattr(self, '_stored_annotations') and self._stored_annotations:
            print(f"Adding {len(self._stored_annotations)} annotations to zoom plot")
            for ann in self._stored_annotations:
                try:
                    ann_x = float(ann.get("x", 0))
                    ann_y = float(ann.get("y", 0))
                    text = str(ann.get("text", ""))
                    is_top = bool(ann.get("is_top", False))
                    
                    # Only add annotation if it's within zoom range
                    if x0 <= ann_x <= x1:
                        # Find corresponding y value in zoom data
                        if np.any(mask):
                            zoom_wl = wl[mask]
                            zoom_intens = intens[mask]
                            
                            # Find closest wavelength point
                            closest_idx = np.argmin(np.abs(zoom_wl - ann_x))
                            if closest_idx < len(zoom_intens):
                                y_pos = zoom_intens[closest_idx]
                                
                                # Create text annotation for zoom plot
                                text_item = pg.TextItem(
                                    text=text,
                                    color=(255, 255, 0) if is_top else (200, 200, 200),  # Yellow for zoom
                                    anchor=(0.5, 1.0)  # Bottom center
                                )
                                text_item.setPos(ann_x, y_pos * 1.1)  # Slightly above peak
                                zoom_widget.addItem(text_item)
                                
                                # Add vertical line for peak in zoom
                                line_item = pg.InfiniteLine(
                                    pos=ann_x,
                                    angle=90,
                                    pen=pg.mkPen(color=(255, 255, 0, 180) if is_top else (200, 200, 200, 120), width=2)
                                )
                                zoom_widget.addItem(line_item)
                                
                                print(f"Added annotation '{text}' at {ann_x:.2f} nm to zoom plot")
                        
                except Exception as e:
                    print(f"Error adding annotation to zoom: {e}")
        else:
            if hasattr(self, '_stored_annotations'):
                print(f"No annotations to add to zoom plot (stored: {len(self._stored_annotations)} annotations)")
            else:
                print("No stored annotations available for zoom plot")

    def export_publication_plot(self):
        """Export zoom region as publication-ready plot"""
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Matplotlib not available for export")
            return
            
        if not hasattr(self, '_data_arrays') or not self._data_arrays:
            QMessageBox.warning(self, "Warning", "No data available for export")
            return
            
        # Get current zoom region
        try:
            if self.roi:
                region = self.roi.getRegion()
                x0, x1 = float(region[0]), float(region[1])  # type: ignore
            else:
                x0, x1 = 300, 800  # Default range
        except:
            # Default to a reasonable range if no zoom
            x0, x1 = 300, 800
            
        # Get filename for export
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Publication Plot",
            f"publication_plot_{x0:.0f}-{x1:.0f}nm.png",
            "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;EPS files (*.eps)"
        )
        
        if not filename:
            return
            
        try:
            self._create_publication_plot(x0, x1, filename)
            QMessageBox.information(self, "Success", f"Plot exported to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export plot:\n{str(e)}")
    
    def _create_publication_plot(self, x0: float, x1: float, filename: str):
        """Create publication-ready plot for zoom region"""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is not available")
            
        import matplotlib.pyplot as plt  # Import here for typing
        
        # Use downsampled data if available for consistency
        if hasattr(self, '_downsampled_data') and self._downsampled_data:
            wl = np.asarray(self._downsampled_data[0])
            intens = np.asarray(self._downsampled_data[1])
            data_source = "Processed (4096 points)"
        elif self._data_arrays:
            wl = np.asarray(self._data_arrays[0])
            intens = np.asarray(self._data_arrays[1])
            data_source = "Original resolution"
        else:
            raise ValueError("No data available to export")
        
        # Filter data to zoom region with some padding
        padding = (x1 - x0) * 0.02  # 2% padding
        mask = (wl >= (x0 - padding)) & (wl <= (x1 + padding))
        zoom_wl = wl[mask]
        zoom_intens = intens[mask]
        
        if len(zoom_wl) == 0:
            raise ValueError("No data points in the selected zoom region")
        
        # Create publication figure with proper size (standard journal column width)
        fig = plt.figure(figsize=(8.5, 6))  # Single column width in inches
        ax = fig.add_subplot(111)
        
        # Plot spectrum with publication styling
        ax.plot(zoom_wl, zoom_intens, 'b-', linewidth=1.5, alpha=0.8, label='Spectrum')
        
        # Add element annotations if available
        annotations_max_y = 0  # Track highest annotation position
        if hasattr(self, '_stored_annotations') and self._stored_annotations:
            annotations_max_y = self._add_publication_annotations(ax, zoom_wl, zoom_intens, x0, x1)
        
        # Set axis labels with proper units
        ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Intensity (a.u.)', fontsize=14, fontweight='bold')
        
        # Get current filename from parent (main window)
        current_filename = "Unknown File"
        parent = self.parent()
        if parent and hasattr(parent, 'current_file_name'):
            current_file = getattr(parent, 'current_file_name', None)
            if current_file and isinstance(current_file, str):
                # Extract just the base filename without extension for cleaner display
                base_name = os.path.splitext(os.path.basename(current_file))[0]
                current_filename = base_name if base_name else current_file
        
        # Set title with filename and region info
        ax.set_title(f'File: {current_filename}\nSpectral Region: {x0:.1f} - {x1:.1f} nm', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Set axis limits with slight padding
        ax.set_xlim(x0 - padding, x1 + padding)
        
        # Calculate intensity range with dynamic padding for annotations
        y_min, y_max = zoom_intens.min(), zoom_intens.max()
        y_range = y_max - y_min
        
        # Adjust y_max to accommodate annotations
        if annotations_max_y > 0:
            # Ensure at least 10% padding above highest annotation
            required_y_max = annotations_max_y + 0.1 * y_range
            y_max_with_padding = max(y_max + 0.1 * y_range, required_y_max)
        else:
            y_max_with_padding = y_max + 0.1 * y_range
            
        ax.set_ylim(y_min - 0.05 * y_range, y_max_with_padding)
        
        # Improve grid and ticks for publication
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1.0, length=3)
        
        # Add minor ticks
        ax.minorticks_on()
        
        # Improve spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')
        
        # Add data source and filename info as small text
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        metadata_text = f'File: {current_filename}\nData: {data_source}\nExported: {timestamp}'
        ax.text(0.02, 0.98, metadata_text, 
               transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', alpha=0.7,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add legend if annotations exist
        if hasattr(self, '_stored_annotations') and self._stored_annotations:
            ax.legend(loc='upper right', frameon=True, fancybox=True, 
                     shadow=True, fontsize=11)
        
        # Tight layout for better spacing
        plt.tight_layout()
        
        # Save with high quality
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none',
                   transparent=False)
        plt.close(fig)
        
        print(f"Publication plot exported: {filename}")
        print(f"Region: {x0:.1f} - {x1:.1f} nm ({len(zoom_wl)} data points)")
    
    def _add_publication_annotations(self, ax, zoom_wl, zoom_intens, x0, x1):
        """Add element annotations to publication plot"""
        if not self._stored_annotations:
            return 0  # Return 0 if no annotations
            
        annotation_colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
        used_positions = []  # Track y-positions to avoid overlap
        max_annotation_y = 0  # Track highest annotation position
        
        for i, annotation in enumerate(self._stored_annotations):
            ann_x = annotation.get('x', 0)
            
            # Only add annotations within zoom region
            if not (x0 <= ann_x <= x1):
                continue
                
            # Find closest wavelength point for intensity
            closest_idx = np.argmin(np.abs(zoom_wl - ann_x))
            if closest_idx < len(zoom_intens):
                ann_y = zoom_intens[closest_idx]
            else:
                continue
                
            # Get element text
            element_text = annotation.get('text', f'Peak {i+1}')
            # Clean up text for publication (remove counts)
            if '(' in element_text and ')' in element_text:
                # Extract just element names, remove counts
                elements = element_text.split(' ')
                clean_elements = []
                for elem in elements:
                    if '(' not in elem:
                        clean_elements.append(elem)
                element_text = ' '.join(clean_elements) if clean_elements else element_text
            
            color = annotation_colors[i % len(annotation_colors)]
            
            # Calculate annotation position to avoid overlap
            y_offset = self._calculate_annotation_offset(ann_y, used_positions, zoom_intens)
            final_annotation_y = ann_y + y_offset
            used_positions.append(final_annotation_y)
            
            # Track maximum annotation height
            max_annotation_y = max(max_annotation_y, final_annotation_y)
            
            # Add vertical line at peak position
            ax.axvline(x=ann_x, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
            
            # Add annotation arrow and text
            ax.annotate(element_text, 
                       xy=(ann_x, ann_y), 
                       xytext=(ann_x, final_annotation_y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.8),
                       fontsize=11, fontweight='bold', color=color,
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                               edgecolor=color, alpha=0.8))
        
        return max_annotation_y  # Return the highest annotation position
    
    def _calculate_annotation_offset(self, base_y, used_positions, intensities):
        """Calculate y-offset for annotation to avoid overlap"""
        y_range = intensities.max() - intensities.min()
        min_offset = y_range * 0.15  # Minimum 15% of range above peak
        
        # Check for overlaps with existing annotations
        for used_y in used_positions:
            if abs(base_y + min_offset - used_y) < y_range * 0.1:  # Too close
                min_offset += y_range * 0.12  # Add more offset
                
        return min_offset
        
        # Add a status label to help debugging
        status_text = f"Zoom region: {x0:.2f} - {x1:.2f} nm with {np.sum(mask)} points ({data_source})"
        print(status_text)
