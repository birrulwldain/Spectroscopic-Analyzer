from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
else:
    # Ini akan diimport di runtime
    from PySide6.QtWidgets import QTableWidgetItem

def update_file_list_display(self):
    """
    Update tampilan daftar file di table widget
    """
    if not hasattr(self, 'files_table') or not hasattr(self, 'asc_files'):
        return
    
    # Hapus semua baris yang ada
    self.files_table.setRowCount(0)
    
    # Tambahkan file ke tabel
    if not self.asc_files:
        return
    
    # Atur jumlah baris sesuai dengan jumlah file
    self.files_table.setRowCount(len(self.asc_files))
    
    # Isi tabel dengan nama file
    for i, filename in enumerate(self.asc_files.keys()):
        item = QTableWidgetItem(filename)
        self.files_table.setItem(i, 0, item)
        
        # Jika file ini adalah file yang aktif, pilih barisnya
        if self.current_file_name == filename:
            self.files_table.selectRow(i)
