from __future__ import annotations
import os
import glob
import traceback
from typing import TYPE_CHECKING, Any, List

import pandas as pd

if TYPE_CHECKING:
    from PySide6.QtCore import QObject, QThread, Signal, Slot
    from PySide6.QtWidgets import (
        QDialog,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QFormLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QFileDialog,
        QProgressBar,
        QTextEdit,
    )
else:
    from PySide6 import QtCore as _QtCore  # type: ignore
    from PySide6 import QtWidgets as _QtWidgets  # type: ignore

    QObject = _QtCore.QObject
    QThread = _QtCore.QThread
    Signal = _QtCore.Signal
    Slot = _QtCore.Slot

    QDialog = _QtWidgets.QDialog
    QWidget = _QtWidgets.QWidget
    QVBoxLayout = _QtWidgets.QVBoxLayout
    QHBoxLayout = _QtWidgets.QHBoxLayout
    QFormLayout = _QtWidgets.QFormLayout
    QLabel = _QtWidgets.QLabel
    QLineEdit = _QtWidgets.QLineEdit
    QPushButton = _QtWidgets.QPushButton
    QFileDialog = _QtWidgets.QFileDialog
    QProgressBar = _QtWidgets.QProgressBar
    QTextEdit = _QtWidgets.QTextEdit

from app.core.analysis import run_full_analysis


class BatchWorker(QObject):
    progress = Signal(int, str)  # percent, message
    finished = Signal(object, str)  # pandas DataFrame, saved_path
    error = Signal(str)

    def __init__(self, input_dir: str, output_dir: str, params: dict[str, Any]):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        # Copy params to avoid accidental mutation by caller
        self.params = dict(params)

    @Slot()
    def run(self):
        try:
            # Scan .asc files recursively
            pattern = os.path.join(self.input_dir, "**", "*.asc")
            files = glob.glob(pattern, recursive=True)
            total = len(files)
            if total == 0:
                self.progress.emit(0, "Tidak ada file .asc ditemukan.")
                df = pd.DataFrame(columns=["Filename", "Predicted Elements", "Element Count", "Peaks Detected"]) 
                saved_path = self._save_summary(df)
                self.finished.emit(df, saved_path)
                return

            rows: List[dict[str, Any]] = []
            encodings = ["utf-8", "utf-16", "latin-1"]

            for idx, file_path in enumerate(files, start=1):
                relname = os.path.relpath(file_path, self.input_dir)
                self.progress.emit(int(idx / total * 100.0), f"Memproses: {relname}")

                text = None
                last_err: Exception | None = None
                for enc in encodings:
                    try:
                        with open(file_path, "r", encoding=enc, errors="strict") as f:
                            text = f.read()
                        break
                    except (UnicodeDecodeError, OSError) as e:  # file encoding or IO error
                        last_err = e
                        continue

                if text is None:
                    # Log row with error info, continue
                    rows.append({
                        "Filename": relname,
                        "Predicted Elements": f"[ERROR: {last_err}]",
                        "Element Count": 0,
                        "Peaks Detected": 0,
                    })
                    continue

                params = dict(self.params)
                params["asc_content"] = text
                # Ensure it's prediction mode
                params["analysis_mode"] = "predict"

                try:
                    results = run_full_analysis(params)
                except (ValueError, RuntimeError, TypeError) as e:
                    rows.append({
                        "Filename": relname,
                        "Predicted Elements": f"[ERROR: {e}]",
                        "Element Count": 0,
                        "Peaks Detected": 0,
                    })
                    continue

                # Extract predicted elements from prediction_table
                pred_table = results.get("prediction_table", []) or []
                elements = [str(r.get("Elemen", "")) for r in pred_table if r.get("Elemen")]
                unique_elements = sorted(set(elements))
                element_str = ", ".join(unique_elements)
                peaks_detected = int(getattr(results.get("peak_wavelengths"), "size", 0) or 0)

                rows.append({
                    "Filename": relname,
                    "Predicted Elements": element_str,
                    "Element Count": len(unique_elements),
                    "Peaks Detected": peaks_detected,
                })

            df = pd.DataFrame(rows)
            saved_path = self._save_summary(df)
            self.progress.emit(100, f"Selesai. Ringkasan disimpan: {saved_path}")
            self.finished.emit(df, saved_path)
        except (RuntimeError, ValueError, OSError, TypeError):
            self.error.emit(traceback.format_exc())

    def _save_summary(self, df: pd.DataFrame) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "batch_summary.xlsx")
        try:
            df.to_excel(out_path, index=False, sheet_name="Ringkasan Batch")
        except (OSError, ValueError):
            # Try CSV fallback
            out_path = os.path.join(self.output_dir, "batch_summary.csv")
            df.to_csv(out_path, index=False)
        return out_path


class BatchDialog(QDialog):
    batchSummaryReady = Signal(object, str)  # DataFrame, saved_path

    def __init__(self, parent: QWidget | None, params_template: dict[str, Any]):
        super().__init__(parent)
        self.setWindowTitle("Proses Folder (Batch)")
        self.setMinimumWidth(520)
        self._params_template = dict(params_template)
        # Remove potential asc_content if passed
        self._params_template.pop("asc_content", None)

        self._thread: QThread | None = None
        self._worker: BatchWorker | None = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.in_edit = QLineEdit()
        self.out_edit = QLineEdit()
        btn_in = QPushButton("Pilih…")
        btn_out = QPushButton("Pilih…")
        row_in = QHBoxLayout(); row_in.addWidget(self.in_edit); row_in.addWidget(btn_in)
        row_out = QHBoxLayout(); row_out.addWidget(self.out_edit); row_out.addWidget(btn_out)
        form.addRow(QLabel("Folder Input:"), row_in)
        form.addRow(QLabel("Folder Output:"), row_out)
        layout.addLayout(form)

        self.progress_bar = QProgressBar(); self.progress_bar.setValue(0)
        self.status_text = QTextEdit(); self.status_text.setReadOnly(True); self.status_text.setMinimumHeight(120)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_text)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Mulai")
        self.btn_close = QPushButton("Tutup")
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        btn_in.clicked.connect(self._choose_input)
        btn_out.clicked.connect(self._choose_output)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_close.clicked.connect(self._on_close)

    def _choose_input(self):
        dirname = QFileDialog.getExistingDirectory(self, "Pilih Folder Input", "")
        if dirname:
            self.in_edit.setText(dirname)

    def _choose_output(self):
        dirname = QFileDialog.getExistingDirectory(self, "Pilih Folder Output", "")
        if dirname:
            self.out_edit.setText(dirname)

    def _on_start(self):
        in_dir = self.in_edit.text().strip()
        out_dir = self.out_edit.text().strip()
        if not in_dir or not os.path.isdir(in_dir):
            self._append_status("Folder input tidak valid.")
            return
        if not out_dir:
            self._append_status("Folder output harus diisi.")
            return

        self.btn_start.setEnabled(False)
        self._append_status(f"Mulai batch:\n- Input: {in_dir}\n- Output: {out_dir}")

        self._thread = QThread(self)
        self._worker = BatchWorker(in_dir, out_dir, self._params_template)
        self._worker.moveToThread(self._thread)

        # Wire signals
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(max(0, min(100, int(pct))))
        self._append_status(msg)

    def _on_finished(self, df: object, saved_path: str):
        # Emit to parent and enable closing
        self._append_status(f"Selesai. Ringkasan: {saved_path}")
        try:
            self.batchSummaryReady.emit(df, saved_path)  # type: ignore[arg-type]
        except (RuntimeError, TypeError):
            pass
        self.btn_start.setEnabled(True)

    def _on_error(self, tb: str):
        self._append_status("Terjadi error:\n" + tb)
        self.btn_start.setEnabled(True)

    def _append_status(self, text: str):
        self.status_text.append(text)

    def _cleanup_thread(self):
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def _on_close(self):
        # If running, let it finish but allow closing dialog
        self.close()
