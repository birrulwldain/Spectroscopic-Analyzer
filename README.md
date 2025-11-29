# Spectroscopic Analyzer
## AI-Powered Laser-Induced Breakdown Spectroscopy (LIBS) Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PySide6](https://img.shields.io/badge/UI-PySide6-green.svg)](https://www.qt.io/qt-for-python)

---

## Overview

**Spectroscopic Analyzer** is a scientific software tool for automated elemental analysis using Laser-Induced Breakdown Spectroscopy (LIBS). The application combines deep learning-based element detection with interactive visualization and publication-ready export functionality.

### Key Features

✨ **Automated Element Detection** — AI-powered identification of 18+ elements from spectral peaks using a multilabel Informer neural network  
📊 **Interactive Analysis** — Drag-to-select wavelength regions and inspect detailed spectral features in real-time  
🔬 **Scientific Rigor** — Baseline correction (Asymmetric Least Squares), peak detection, and statistical validation  
💾 **Batch Processing** — Process entire folders of spectra with saved parameter presets  
🎨 **Publication-Ready Plots** — Export high-resolution figures (300 DPI) with customizable labels  
⚙️ **Cross-Platform** — Windows, macOS, and Linux support via Python and Qt

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/spectroscopic-analyzer.git
cd spectroscopic-analyzer

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
python app/main.py
```

The GUI will open with four main panels:
- **Upper Left**: Full spectrum visualization with drag-select region
- **Upper Right**: Detailed view of selected region with element labels
- **Lower Left**: Parameter control and batch operations
- **Lower Right**: Analysis results and element detection table

### Basic Workflow

1. **Load Data**: "📂 Load Folder" or "📄 Load File" to select `.asc` spectroscopic data
2. **Preprocess**: "🔧 Preprocess" to normalize and apply baseline correction
3. **Predict**: "🤖 Predict" to run element detection
4. **Inspect**: Drag on main plot to zoom into regions of interest
5. **Export**: "📊 Export Scientific Plot" for publication-ready figures

---

## Project Structure

```
spectroscopic-analyzer/
├── app/                            # Application source code
│   ├── main.py                     # Entry point (launches GUI)
│   ├── model.py                    # Deep learning model (PyTorch)
│   ├── processing.py               # Data preprocessing utilities
│   ├── core/
│   │   └── analysis.py             # Complete analysis pipeline
│   └── ui/
│       ├── main_window.py          # Main GUI window
│       ├── control_panel.py        # Parameter controls
│       ├── results_panel.py        # Results visualization
│       ├── batch_dialog.py         # Batch processing dialog
│       └── worker.py               # Background processing threads
├── assets/                         # Model and data assets
│   ├── informer_multilabel_model.pth
│   ├── element-map-18a.json
│   └── wavelengths_grid.json
├── example-asc/                    # Example LIBS spectroscopic data files
├── tests/                          # Unit tests
├── docs/
│   ├── ARCHITECTURE.md             # Technical architecture
│   ├── API.md                      # Python API reference
│   └── openapi.yaml                # REST API spec (if deployed)
├── README.md                       # This file
├── CONTRIBUTING.md                 # Contribution guidelines
├── TROUBLESHOOTING.md              # Common issues and solutions
├── LICENSE                         # MIT License
├── requirements.txt                # Python dependencies
└── Dockerfile                      # Container image
```

---

## Documentation

### User Guide
- **Installation & Setup**: See [Quick Start](#quick-start) section
- **Parameter Reference**: [docs/PARAMETERS.md](docs/PARAMETERS.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Example Workflows**: [docs/EXAMPLES.md](docs/EXAMPLES.md)

### Developer Guide
- **Architecture Overview**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Python API Reference**: [docs/API.md](docs/API.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Development Setup**: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

---

## Core Components

### Analysis Pipeline (`app/core/analysis.py`)

```python
result = run_full_analysis(input_data)
```

Main function that processes spectroscopic data through:
- **Parsing**: Read ASC (ASCII) format spectral data
- **Preprocessing**: Baseline correction using Asymmetric Least Squares (ALS)
- **Peak Detection**: Automatic peak identification with configurable thresholds
- **Element Assignment**: Map peaks to elements using learned wavelength mapping
- **Validation**: Statistical metrics and confidence scores
- **Abel Deconvolution** (optional): For cylindrical sample geometry

**Key Parameters:**
- `baseline_lambda`: Smoothness parameter for ALS baseline (default: 1e5)
- `target_max_intensity`: Normalization target (default: 0.8)
- `peak_height`: Minimum peak height (default: 0.01)
- `peak_prominence`: Minimum peak prominence (default: 0.01)
- `max_peaks`: Maximum peaks to detect (default: 50)

### Neural Network Model (`app/model.py`)

- **Architecture**: Multilabel Informer network (PyTorch)
- **Input**: Normalized spectral data (interpolated to 18000+ wavelength points)
- **Output**: 18 element probabilities (0-1 confidence scores)
- **Training Data**: LIBS spectra from diverse elemental standards

**Key Functions:**
```python
model = load_model()  # Load pretrained weights
pred = model(spectrum)  # Predict elements [batch, 18]
baseline = als_baseline_correction(spectrum, lam=1e5, p=0.001, niter=10)
```

### Data Processing (`app/processing.py`)

```python
normalized_spectrum = prepare_asc_data(
    asc_content_string,
    target_wavelengths,
    target_max_intensity=0.8,
    als_lambda=1e5,
    als_p=0.001,
    als_max_iter=10
)
```

---

## Data Formats

### Input: ASC (ASCII Spectroscopy) Files

Standard two-column format (space-separated):

```
# Wavelength (nm)    Intensity (counts)
250.5               1234
251.2               5678
252.1               9234
...
750.0               2341
```

### Output Files

**CSV/Excel Results:**
- Element names and probabilities
- Peak positions and intensities
- Spectral metrics (SNR, baseline deviation, etc.)
- Timestamps and parameter log

**Exported Figures (PNG):**
- 300 DPI resolution (publication-ready)
- Element labels at detected peak positions
- Wavelength range specification in filename
- PDF vector format available on request

---

## System Requirements

### Minimum
- **OS**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python**: 3.9+
- **RAM**: 4 GB
- **Disk**: 2 GB (including model weights)

### Recommended
- **RAM**: 8+ GB
- **GPU**: NVIDIA CUDA 11.0+ (optional, for faster inference)
- **Storage**: SSD for better file I/O

### Python Dependencies
- PyTorch 2.0+
- PySide6 (Qt6 bindings)
- PyQtGraph (scientific visualization)
- NumPy, SciPy, Pandas
- PyAbel (optional, for deconvolution)

See [requirements.txt](requirements.txt) for complete versions.

---

## Citation & Publication

If you use Spectroscopic Analyzer in your research, please cite:

```bibtex
@software{spectroscopic_analyzer_2025,
  author = {Birrulwaldi Nurdin},
  title = {Spectroscopic Analyzer: AI-Powered LIBS Analysis Software},
  year = {2025},
  url = {https://github.com/yourusername/spectroscopic-analyzer},
  note = {v1.0.0}
}
```

### References

Key papers and resources:

1. **Informer Architecture**: Zhouxianghui et al. (2020). "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." arXiv:2012.07436

2. **Baseline Correction**: Eilers & Boelens (2005). "Baseline correction with asymmetric least squares smoothing." Technical report, Leiden University.

3. **LIBS Analysis**: Cremers & Radziemski (2006). Handbook of Laser-Induced Breakdown Spectroscopy.

4. **Abel Deconvolution**: Dribinski et al. (2015). "The Velocity Map Imaging technique." Rev. Sci. Instrum. 86, 033103.

---

## Troubleshooting

### Application crashes on startup
```
AttributeError: 'MainWindow' object has no attribute 'X'
```
**Solution:** Clear cache and reinstall
```bash
find . -type d -name __pycache__ -exec rm -rf {} +
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### CUDA/GPU not detected
Application runs on CPU by default. For GPU acceleration:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Import errors
```bash
# Verify virtual environment
which python
python -c "import torch, PySide6, pyqtgraph; print('OK')"
```

For detailed help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open an [Issue](../../issues).

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style and quality standards
- Testing requirements
- Pull request process
- Issue reporting guidelines

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

Redistribution and use in source and binary forms permitted with attribution.

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features:
- Sprint 1: Batch processing & parameter presets
- Sprint 2: NIST database integration & peak fitting
- Sprint 3: Quantitative analysis & PDF reporting

---

## Contact & Support

- **Issues & Bugs**: [GitHub Issues](../../issues)
- **Discussions & Ideas**: [GitHub Discussions](../../discussions)
- **Email**: birrulwaldi@example.com

---

## Acknowledgments

- Deep learning model based on Informer architecture
- Baseline correction via Asymmetric Least Squares method
- Abel deconvolution via [PyAbel](https://github.com/PyAbel/PyAbel)
- Qt framework via [PySide6](https://wiki.qt.io/Qt_for_Python)

---

**Last Updated**: November 29, 2025  
**Version**: 1.0.0-beta  
**Status**: Active development

