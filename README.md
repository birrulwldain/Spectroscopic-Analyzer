# Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status: In Press](https://img.shields.io/badge/Status-In%20Press-blue.svg)]()

---

## Overview

This repository contains the implementation and experimental data accompanying the paper:

> **"Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine"**  
> Walidain, B., Idris, N., Saddami, K., Yuzza, N., & Mitaphonna, R. (2025)  
> *IOP Conference Series: Earth and Environmental Science*, AIC 2025

The work presents an Informer-based deep learning model for qualitative multi-element analysis via Laser-Induced Breakdown Spectroscopy (LIBS). The model is trained on physics-based synthetic spectra generated using the Saha–Boltzmann equation and evaluated on an experimental case study of Aceh traditional herbal medicine samples. The implementation includes training and inference scripts, as well as an interactive GUI for spectroscopic data analysis.

### Key Features

- **Physics-Based Synthetic Spectral Library** — Training spectra generated via Saha–Boltzmann plasma theory for robust multi-element representation
- **Informer Encoder Architecture** — 2-layer ProbSparse attention mechanism for efficient processing of 4096-channel high-resolution spectra
- **Multi-Label Classification** — Simultaneous detection of 17 elements + background class from a single LIBS spectrum
- **Experimental Case Study** — Qualitative analysis of Aceh traditional herbal medicine samples
- **Reproducible Workflow** — Complete scripts for model training, evaluation, and inference with documented hyperparameters
- **Interactive GUI** — PySide6-based graphical interface for real-time spectral visualization and element identification

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/birrulwaldain/informer-libs-multielement.git
cd informer-libs-multielement

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Interactive GUI

```bash
python app/main.py
```

The GUI displays four main panels:
- **Upper Left**: Full spectrum visualization with drag-select region
- **Upper Right**: Detailed view of selected region with element labels
- **Lower Left**: Parameter control and batch operations
- **Lower Right**: Analysis results and element detection table

### Basic Workflow

1. **Load Data**: Use "Load Folder" or "Load File" to select .asc spectroscopic data
2. **Preprocess**: Click "Preprocess" to normalize and apply baseline correction
3. **Predict**: Click "Predict" to run element detection
4. **Inspect**: Drag on main plot to zoom into regions of interest
5. **Export**: Click "Export Scientific Plot" for publication-ready figures

---

## Reproducibility & Training

For step-by-step instructions on reproducing the training and evaluation results from the paper, see **[REPRODUCE.md](REPRODUCE.md)**.

### Core Scripts Summary

The `scripts/` directory contains:

- **train.py** — Train the Informer model using all default hyperparameters from the paper
- **eval.py** — Evaluate trained model and generate per-element metrics table (matching Table perf_detail in paper)
- **check.py** — Verify dataset integrity before training
- **planner.py, job.py, merge.py** — Generate synthetic LIBS spectra via Saha–Boltzmann equations

For HPC cluster deployments (tested on BRIN's Mahameru cluster):
- **job.sh** — SLURM job submission script (requires adaptation for other clusters)
- **run.sh, run-eval.sh, job-merge.sh** — Local execution wrappers

For complete documentation of all scripts, see **[scripts/README.md](scripts/README.md)**.

---

## Repository Structure

```
informer-libs-multielement/
├── app/                            # Application source code
│   ├── main.py                     # GUI application entry point
│   ├── model.py                    # Informer model and utility functions
│   ├── processing.py               # Spectral data preprocessing
│   ├── core/
│   │   └── analysis.py             # Complete analysis pipeline
│   └── ui/
│       ├── main_window.py          # Main application window
│       └── ...
├── assets/                         # Model weights and reference data
│   ├── informer_multilabel_model.pth    # Pretrained model weights
│   ├── element-map-17.json              # Element-wavelength mapping
│   └── wavelengths_grid.json            # Target wavelength grid (4096 channels)
├── data/                           # Training and experimental data
│   ├── synthetic/                  # Synthetic training data (HDF5)
│   ├── experimental/               # Real LIBS measurements (on request)
│   └── README.md                   # Data documentation
├── example-asc/                    # Example LIBS spectra for testing
├── scripts/                        # Training, evaluation, and utility scripts
│   └── README.md                   # Script documentation
├── docs/                           # Technical documentation
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT.md
│   ├── PARAMETERS.md
│   └── openapi.yaml
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup configuration
├── pyproject.toml                  # Modern Python packaging
├── REPRODUCE.md                    # Complete reproducibility guide
├── MODEL_CARD.md                   # Model documentation card
├── LICENSE                         # MIT License
├── CITATION.cff                    # Citation metadata
├── README.md                       # This file
└── ROADMAP.md                      # Future plans and improvements
```

---

## Data

### Synthetic Training Data

The model is trained on synthetic LIBS spectra generated via the Saha–Boltzmann plasma equations. Data generation uses the scripts in `scripts/` and can be reproduced from scratch.

**Dataset Statistics:**
- Training: 8,000 spectra
- Validation: 1,000 spectra
- Test: 1,000 spectra
- Elements: 17 + background
- Wavelength range: 200–850 nm (4096 channels)
- Noise levels: 1%, 2%, 5%

For detailed data documentation, see **[data/README.md](data/README.md)**.

### Experimental Data

Real LIBS measurements from Aceh traditional herbal medicine samples are not included in the public repository due to privacy considerations. Access can be requested from the corresponding author.

### Example Data

Sample spectra are provided in `example-asc/` for quick testing without requiring data generation.

---

## Model Card

For technical details about the Informer model architecture, training configuration, performance metrics, and intended use, see **[MODEL_CARD.md](MODEL_CARD.md)**.

Key specifications:
- Architecture: Informer encoder with 2 layers
- Embedding dimension: 32
- Feed-forward dimension: 64
- Attention heads: 4
- Output: Multi-label probabilities (0-1 per element)
- Trained on: Synthetic LIBS spectra via Saha–Boltzmann equations

---

## Citation

If you use this code, data, or model in your research, please cite the paper:

### BibTeX Entry

```bibtex
@inproceedings{Walidain2025,
  title={Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine},
  author={Walidain, Birrul and Idris, Nasrullah and Saddami, Khairun and Yuzza, Natasya and Mitaphonna, Rara},
  booktitle={AIC 2025 -- Natural Life and Sciences track},
  journal={IOP Conference Series: Earth and Environmental Science},
  year={2025},
  note={in press}
}
```

### Human-Readable Citation

Walidain, B., Idris, N., Saddami, K., Yuzza, N., & Mitaphonna, R. (2025). Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine. *IOP Conference Series: Earth and Environmental Science*, AIC 2025. (in press)

See **[CITATION.cff](CITATION.cff)** for additional citation formats (CFF, RIS, Zotero).

---

## Acknowledgments

This work builds upon:
- The **Informer architecture** from Zhou et al. (ICLR 2021)
- **ProbSparse attention** mechanism for efficient sequence processing
- **Saha–Boltzmann physics** for synthetic spectral generation
- **Asymmetric Least Squares** for baseline correction
- **PyTorch, PySide6, PyQtGraph** for the implementation stack

We also acknowledge:
- BRIN (Badan Riset dan Inovasi Nasional) for computing resources
- Department of Physics, Universitas Syiah Kuala for experimental facilities

---

## Contact & Support

**Corresponding Author:**
- Name: Nasrullah Idris
- Email: nasrullah.idris@usk.ac.id
- Affiliation: Department of Physics, Faculty of Mathematics and Natural Sciences, Universitas Syiah Kuala, Banda Aceh 23111, Indonesia

**GitHub Maintainer:**
- Name: Birrul Walidain
- Email: birrul@mhs.usk.ac.id
- Repository: [github.com/birrulwaldain/informer-libs-multielement](https://github.com/birrulwaldain/informer-libs-multielement)

**For Issues & Questions:**
- Open an issue: [GitHub Issues](../../issues)
- Email: birrul@mhs.usk.ac.id

---

## Contributing

Contributions are welcome. Please see **[CONTRIBUTING.md](CONTRIBUTING.md)** for:
- Code style and quality standards
- Testing requirements
- Pull request process
- Issue reporting guidelines

---

## License

This project is licensed under the **MIT License** — see **[LICENSE](LICENSE)** for details.

Redistribution and use in source and binary forms are permitted with attribution.

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

The application runs on CPU by default. For GPU acceleration:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Import errors

```bash
# Verify virtual environment
which python
python -c "import torch, PySide6, pyqtgraph; print('OK')"
```

For additional help, see **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**.

---

## Roadmap

See **[ROADMAP.md](ROADMAP.md)** for planned features and improvements:
- Batch processing and parameter presets
- NIST spectral database integration
- Peak fitting and quantitative analysis
- PDF report generation

---

## Related Documentation

- **[REPRODUCE.md](REPRODUCE.md)** — Complete reproducibility guide
- **[MODEL_CARD.md](MODEL_CARD.md)** — Model technical card
- **[data/README.md](data/README.md)** — Data documentation and format
- **[scripts/README.md](scripts/README.md)** — Script documentation
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Contribution guidelines
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Application architecture
- **[docs/PARAMETERS.md](docs/PARAMETERS.md)** — Model parameters reference

---

**Last Updated:** November 30, 2025  
**Version:** 1.0.0  
**Status:** Ready for Publication

