# Tambahkan style sheet global untuk ControlPanel
CONTROL_PANEL_STYLE = """
QGroupBox {
    font-weight: bold;
    font-size: 10pt;
    border: 1px solid #cccccc;
    border-radius: 5px;
    margin-top: 1ex;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: #0066cc;
}
QLabel {
    font-size: 9pt;
}
QLineEdit, QComboBox, QDoubleSpinBox {
    padding: 4px;
    border: 1px solid #cccccc;
    border-radius: 3px;
    background-color: #ffffff;
}
QPushButton {
    padding: 5px;
    border-radius: 4px;
}
"""
