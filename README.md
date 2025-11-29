# Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status: In Press](https://img.shields.io/badge/Status-In%20Press-blue.svg)]()

---

## Overview

This repository contains the implementation and experimental data accompanying the paper **"Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine"** (to appear in *IOP Conference Series: Earth and Environmental Science*, AIC 2025). 

The work presents an Informer-based deep learning model for qualitative multi-element analysis via Laser-Induced Breakdown Spectroscopy (LIBS). The model is trained on physics-based synthetic spectra generated using the Saha–Boltzmann equation and evaluated on an experimental case study of Aceh traditional women's medicine. The implementation includes training and inference scripts, as well as an interactive GUI for spectroscopic data analysis.

### Key Features

**Physics-Based Synthetic Spectral Library** — Training spectra generated via Saha–Boltzmann plasma theory for robust multi-element representation  
**Informer Encoder Architecture** — 2-layer ProbSparse attention mechanism for efficient processing of 4096-channel high-resolution spectra  
**Multi-Label Classification** — Simultaneous detection of 17 elements + background class from a single LIBS spectrum  
**Experimental Case Study** — Qualitative analysis of Aceh traditional women's medicine samples  
**Reproducible Workflow** — Complete scripts for model training, evaluation, and inference with documented hyperparameters  
**Interactive GUI** — PySide6-based graphical interface for real-time spectral visualization and element identification

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

1. **Load Data**: "Load Folder" or "Load File" to select `.asc` spectroscopic data
2. **Preprocess**: "Preprocess" to normalize and apply baseline correction
3. **Predict**: "Predict" to run element detection
4. **Inspect**: Drag on main plot to zoom into regions of interest
5. **Export**: "Export Scientific Plot" for publication-ready figures

---

## Training & HPC Pipeline (Reproducibility)

The full training and evaluation workflow used in the paper is implemented via the scripts in the `scripts/` directory. These scripts are **not required** to run the GUI, but are provided for reproducibility and for users who want to retrain or extend the model.

### Core Scripts (Training & Evaluation)

- `scripts/train.py`  
  Training pipeline for the Informer-based multi-element LIBS classifier. It loads pre-generated synthetic LIBS spectra (Saha–Boltzmann based), trains a 2-layer Informer encoder (default configuration as reported in the paper), and saves model checkpoints.

- `scripts/eval.py`  
  Evaluation pipeline for a trained model. It loads a trained checkpoint, runs inference on the evaluation split, and computes multi-label metrics such as accuracy, Hamming loss, and F1-score. This script is used to derive the **per-element precision/recall/F1 table** (Table `perf_detail` in the paper). Results can be further exported (e.g. to CSV/Excel) from this script.

- `scripts/check.py`  
  Dataset integrity checker for the HDF5 datasets used for training and evaluation. It verifies that all expected elements and splits are present and that tensor shapes are consistent. This script is useful when preparing or debugging large synthetic datasets.

### HPC-Oriented Scripts (Mahameru / Cluster BRIN)

The following scripts were written for running large-scale synthetic data generation and training on a GPU cluster (e.g. BRIN's Mahameru cluster). They may require adaptation of paths and scheduler directives on other systems.

- `scripts/planner.py`  
  Job planner for HPC runs. Generates a list of simulation or training jobs (e.g. combinations of plasma parameters, compositions, noise levels) to be dispatched to the cluster scheduler.

- `scripts/job.py`  
  Per-node worker script intended to run on each HPC node. It consumes a planned job configuration, generates or processes LIBS spectra for that job, and writes partial results (e.g. HDF5 shards).

- `scripts/job.sh`  
  SLURM batch submission script used on BRIN's Mahameru cluster. It configures resources (partition, CPUs, memory, walltime), activates the appropriate Conda environment, and then calls `job.py` with a specific JSON chunk (e.g. `combinations-<suffix>-<chunk>.json`) to produce a corresponding HDF5 shard (e.g. `dataset-<suffix>-<chunk>.h5`). This script is **cluster-specific** and should be adapted (paths, `#SBATCH` directives, environment activation) for other HPC systems.

- `scripts/merge.py`  
  Merge utility for combining multiple HDF5 shards produced by `job.py` into a single consolidated dataset file (with `train/`, `validation/`, and `test` splits). This merged dataset is then used as input to `scripts/train.py`.

- `scripts/map.py` and `scripts/conv.py`  
  Helper scripts for mapping, conversion, or reshaping datasets and labels between different intermediate formats. These are mainly internal research tools; they are **not required** to reproduce the main results of the paper, but are useful when regenerating datasets from raw simulation outputs.

> **Note:** The HPC scripts (`planner.py`, `job.py`, `job.sh`, `merge.py`) were originally tailored to a specific cluster environment and may contain assumptions about file system layout, job scheduling, or data locations. When porting them to a different cluster, adapt the scheduler directives and paths accordingly.

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
│   └── ui/                         # GUI components (PySide6)
│       ├── main_window.py          # Main application window
│       └── ...
├── assets/                         # Model weights and reference data
│   ├── informer_multilabel_model.pth    # Pretrained model weights
│   ├── element-map-17.json              # Element-wavelength mapping
│   └── wavelengths_grid.json            # Target wavelength grid (4096 channels)
├── data/                           # Experimental and synthetic data
│   ├── synthetic/                  # Training data (Saha–Boltzmann spectra)
│   └── experimental/               # Case study measurements
├── models/                         # Saved checkpoints and model definitions
├── notebooks/                      # Jupyter notebooks for analysis
├── training/                       # Training scripts
├── scripts/                        # Utility and inference scripts
├── tests/                          # Unit tests
├── docs/                           # Technical documentation
│   ├── ARCHITECTURE.md
│   └── ...
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── README.md                       # This file
├── LICENSE                         # MIT License
└── CITATION.cff                    # Citation metadata
```

---

## How to Cite

If you use this code or data in your research, please cite the accompanying paper:

### Human-Readable Citation

Walidain, B., Idris, N., Saddami, K., Yuzza, N., & Mitaphonna, R. (2025). Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine. *IOP Conference Series: Earth and Environmental Science*, AIC 2025. doi: to be assigned

### BibTeX

```bibtex
@inproceedings{Walidain2025,
  title={Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine},
  author={Walidain, Birrul and Idris, Nasrullah and Saddami, Khairun and Yuzza, Natasya and Mitaphonna, Rara},
  booktitle={AIC 2025 -- Natural Life and Sciences track},
  journal={IOP Conference Series: Earth and Environmental Science},
  year={2025},
  doi = {to be assigned},
  note={in press}
}
```

---

## Contact

**Corresponding Author:**
- **Name**: Nasrullah Idris
- **Email**: [nasrullah.idris@usk.ac.id](mailto:nasrullah.idris@usk.ac.id)
- **Affiliation**: Department of Physics, Faculty of Mathematics and Natural Sciences, Universitas Syiah Kuala, Banda Aceh 23111, Indonesia

**GitHub Maintainer:**
- **Name**: Birrul Walidain
- **Email**: [birrul@mhs.usk.ac.id](mailto:birrul@mhs.usk.ac.id)
- **Repository**: [github.com/birrulwaldain/informer-libs-multielement](https://github.com/birrulwaldain/informer-libs-multielement)

---

## Citation & Publication

If you use this implementation in your research, please cite the paper above. The BibTeX entry will be updated with the DOI once assigned by IOP Publishing.

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
- **Email**: birrul@mhs.usk.ac.id

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
