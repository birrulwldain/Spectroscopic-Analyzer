from __future__ import annotations
from typing import TYPE_CHECKING, Any

import numpy as np
import pyqtgraph as pg

if TYPE_CHECKING:
    from PySide6.QtCore import Slot, QSize
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
    )
else:
    from PySide6 import QtCore as _QtCore  # type: ignore
    from PySide6 import QtWidgets as _QtWidgets  # type: ignore

    Slot = _QtCore.Slot
    QSize = _QtCore.QSize

    QWidget = _QtWidgets.QWidget
    QVBoxLayout = _QtWidgets.QVBoxLayout
    QHBoxLayout = _QtWidgets.QHBoxLayout
    QGroupBox = _QtWidgets.QGroupBox
    QTableWidget = _QtWidgets.QTableWidget
    QTableWidgetItem = _QtWidgets.QTableWidgetItem
    QHeaderView = _QtWidgets.QHeaderView
    QPushButton = _QtWidgets.QPushButton
    QCheckBox = _QtWidgets.QCheckBox
    QSizePolicy = _QtWidgets.QSizePolicy


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
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.roi = None
        self._zoom_curve = None
        # (wmin, wmax, ymin, ymax)
        self._data_bounds = None
        # Guard to avoid recursive ROI updates
        self._roi_updating = False
        # Track whether we've applied the default ROI for the current data
        self._roi_initialized = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        plot_group = QGroupBox("Hasil Interpretasi Grafik")
        plot_layout = QVBoxLayout(plot_group)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setTitle("Plot Utama (Overview)")
        plot_layout.addWidget(self.plot_widget, 3)

        self.zoom_plot_widget = SquarePlotWidget()
        self.zoom_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.zoom_plot_widget.setTitle("Plot Zoom (Area Terpilih)")
        try:
            self.zoom_plot_widget.getViewBox().setAspectLocked(True)
        except (RuntimeError, AttributeError):
            pass
        plot_layout.addWidget(self.zoom_plot_widget, 2)

        self.radial_plot_widget = pg.PlotWidget()
        self.radial_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.radial_plot_widget.setTitle("Profil Radial (Abel)")
        plot_layout.addWidget(self.radial_plot_widget, 1)

        plot_controls_layout = QHBoxLayout()
        self.zoom_mode_checkbox = QCheckBox("Tampilkan Seleksi Zoom")
        self.zoom_mode_checkbox.setChecked(True)
        plot_controls_layout.addWidget(self.zoom_mode_checkbox)
        plot_controls_layout.addStretch()
        self.btn_reset = QPushButton("Reset View")
        plot_controls_layout.addWidget(self.btn_reset)
        plot_layout.addLayout(plot_controls_layout)

        table_group = QGroupBox("Tabel Hasil")
        table_layout = QVBoxLayout(table_group)
        self.table_widget = QTableWidget()
        table_layout.addWidget(self.table_widget)

        layout.addWidget(plot_group, 2)
        layout.addWidget(table_group, 1)

        self.zoom_mode_checkbox.toggled.connect(self.on_zoom_toggle)
        self.btn_reset.clicked.connect(self.on_reset_clicked)

    def on_region_changed(self):
        if self.roi is None:
            return
        try:
            pos = self.roi.pos()
            size = self.roi.size()
            x0 = float(pos.x()); y0 = float(pos.y())
            w = float(size.x()); h = float(size.y())
            x1 = x0 + w; y1 = y0 + h
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0

            # Clamp to data bounds so the zoom view never goes out of range
            eps = 1e-9
            if self._data_bounds is not None:
                wmin, wmax, ymin, ymax = self._data_bounds
                minw = max(eps, 0.001 * max(eps, wmax - wmin))
                minh = max(eps, 0.001 * max(eps, ymax - ymin))
                ox0, oy0, ow, oh = x0, y0, (x1 - x0), (y1 - y0)
                if (x1 - x0) < minw:
                    x1 = x0 + minw
                if (y1 - y0) < minh:
                    y1 = y0 + minh
                x0 = max(wmin, min(x0, wmax - minw))
                x1 = min(wmax, max(x1, wmin + minw))
                y0 = max(ymin, min(y0, ymax - minh))
                y1 = min(ymax, max(y1, ymin + minh))
                nx, ny = x0, y0
                nw, nh = (x1 - x0), (y1 - y0)
                if (abs(nx - ox0) > eps) or (abs(ny - oy0) > eps) or (abs(nw - ow) > eps) or (abs(nh - oh) > eps):
                    try:
                        if not self._roi_updating:
                            self._roi_updating = True
                            try:
                                self.roi.blockSignals(True)
                                self.roi.setPos(nx, ny)
                                self.roi.setSize([nw, nh])
                            finally:
                                self.roi.blockSignals(False)
                                self._roi_updating = False
                    except Exception:
                        pass

            vbz = self.zoom_plot_widget.getViewBox()
            vbz.setXRange(x0, x1)
            vbz.setYRange(y0, y1)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return

    @Slot(dict)
    def update_preview(self, results: dict[str, Any]):
        wl = results.get("wavelengths")
        y = results.get("spectrum_data")

        self.plot_widget.clear()
        self.zoom_plot_widget.clear()
        self.radial_plot_widget.clear()

        self.table_widget.clear()
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(0)

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
                # Reset default-ROI flag for a fresh dataset
                self._roi_initialized = False
                try:
                    self._zoom_curve = self.zoom_plot_widget.plot(wl_arr, y_arr, pen=pg.mkPen('b', width=1.5))
                except (RuntimeError, TypeError, ValueError):
                    pass

                wmin, wmax = float(np.nanmin(wl_arr)), float(np.nanmax(wl_arr))
                ymin, ymax = float(np.nanmin(y_arr)), float(np.nanmax(y_arr))
                self._data_bounds = (wmin, wmax, ymin, ymax)

                # Prepare ROI if needed and set its default to 200-900 nm (clamped to data)
                if self.roi is None:
                    try:
                        self._create_default_roi(wl_arr, y_arr, wmin, wmax, ymin, ymax)
                        self.roi.sigRegionChanged.connect(self.on_region_changed)
                    except (RuntimeError, TypeError, ValueError):
                        self.roi = None
                # Apply default ROI only once per dataset
                if not self._roi_initialized and self.roi is not None:
                    try:
                        self._set_default_roi(wl_arr, y_arr, wmin, wmax, ymin, ymax)
                        self._roi_initialized = True
                    except Exception:
                        pass

                if self.zoom_mode_checkbox.isChecked() and self.roi is not None:
                    try:
                        if self.roi.scene() is None:
                            self.plot_widget.addItem(self.roi)
                    except (RuntimeError, AttributeError):
                        pass

                self.on_region_changed()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass

    def _preferred_xrange(self, wmin: float, wmax: float) -> tuple[float, float]:
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
        y0 = ymin + 0.05 * dy
        roi_w = max(1e-6, x1 - x0)
        roi_h = max(1e-6, 0.9 * dy)
        self.roi = pg.RectROI([x0, y0], [roi_w, roi_h], pen=pg.mkPen('r', width=1.5))
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addScaleHandle([0, 0], [1, 1])

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
        y0 = ymin + 0.05 * dy
        roi_w = max(1e-6, x1 - x0)
        roi_h = max(1e-6, 0.9 * dy)
        try:
            self.roi.blockSignals(True)
            self.roi.setPos(x0, y0)
            self.roi.setSize([roi_w, roi_h])
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
            self.table_widget.setRowCount(len(table_data))
            self.table_widget.setColumnCount(len(headers))
            self.table_widget.setHorizontalHeaderLabels(headers)
            for i, row in enumerate(table_data):
                for j, key in enumerate(headers):
                    self.table_widget.setItem(i, j, QTableWidgetItem(str(row.get(key, ""))))
            self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
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
            vb2 = self.zoom_plot_widget.getViewBox()
            vb2.autoRange()
            try:
                vb2.setAspectLocked(True)
            except (RuntimeError, AttributeError):
                pass
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
