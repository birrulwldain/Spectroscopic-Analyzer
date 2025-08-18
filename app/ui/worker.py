from __future__ import annotations
import traceback
from typing import Any

# Runtime-safe Qt imports to avoid type stub issues
from PySide6 import QtCore as _QtCore  # type: ignore
QObject = _QtCore.QObject
Signal = _QtCore.Signal
Slot = _QtCore.Slot

from app.core.analysis import run_full_analysis


class Worker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    @Slot(dict)
    def run_analysis(self, input_data: dict[str, Any]):
        try:
            results = run_full_analysis(input_data)
            self.finished.emit(results)
        except (ValueError, RuntimeError) as e:
            self.error.emit(f"Error: {e}\n{traceback.format_exc()}")
