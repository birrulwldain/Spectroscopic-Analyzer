from __future__ import annotations
from typing import TYPE_CHECKING, Any

import numpy as np
import pyqtgraph as pg

if TYPE_CHECKING:
    from PySide6.QtCore import Slot, QSize, Signal, Qt, QTimer
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
        QScrollArea,
        QLabel,
    )
else:
    from PySide6 import QtCore as _QtCore  # type: ignore
    from PySide6 import QtWidgets as _QtWidgets  # type: ignore

    Slot = _QtCore.Slot
    Signal = _QtCore.Signal
    QSize = _QtCore.QSize
    Qt = _QtCore.Qt
    QTimer = _QtCore.QTimer

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
        self.update_preview(results)

        table_data = (
            results.get("validation_table") if results.get("validation_table") else results.get("prediction_table", [])
        )
        if not table_data:
            return
        try:
            headers = list(table_data[0].keys())
            # Removed table_widget references
        except (TypeError, ValueError, AttributeError, IndexError):
            pass

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
        zoom_widget.setTitle(f"ZOOM: {x0:.2f} - {x1:.2f} nm")
        print(f"Plotting data in zoom: {np.sum(mask)} points in range")
        
        # Use a very bright color with thicker line
        bright_pen = pg.mkPen(color=(255, 0, 0), width=3)
        plot_item = zoom_widget.plot(wl[mask], intens[mask], pen=bright_pen)
        print(f"Zoom plot created with data: {plot_item is not None}")
        
        # Add a status label to help debugging
        status_text = f"Zoom region: {x0:.2f} - {x1:.2f} nm with {np.sum(mask)} points"
        print(status_text)
        
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
