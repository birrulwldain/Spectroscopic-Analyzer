# DETAILED DIFFS - Metadata Curation Changes

## File 1: README.md

### Change 1: Title and Overview

```diff
- # Spectroscopic Analyzer
- ## AI-Powered Laser-Induced Breakdown Spectroscopy (LIBS) Analysis
+ # Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine

- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
- [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
- [![PySide6](https://img.shields.io/badge/UI-PySide6-green.svg)](https://www.qt.io/qt-for-python)
+ [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
+ [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
+ [![Status: In Press](https://img.shields.io/badge/Status-In%20Press-blue.svg)]()

- ## Overview
- 
- **Spectroscopic Analyzer** is a scientific software tool for automated elemental analysis using Laser-Induced Breakdown Spectroscopy (LIBS). The application combines deep learning-based element detection with interactive visualization and publication-ready export functionality.
+ ## Overview
+ 
+ This repository contains the implementation and experimental data accompanying the paper **"Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine"** (to appear in *IOP Conference Series: Earth and Environmental Science*, AIC 2025).
+ 
+ The work presents an Informer-based deep learning model for qualitative multi-element analysis via Laser-Induced Breakdown Spectroscopy (LIBS). The model is trained on physics-based synthetic spectra generated using the Saha–Boltzmann equation and evaluated on an experimental case study of Aceh traditional women's medicine. The implementation includes training and inference scripts, as well as an interactive GUI for spectroscopic data analysis.
```

### Change 2: Key Features

```diff
- ### Key Features
- 
- ✨ **Automated Element Detection** — AI-powered identification of 18+ elements from spectral peaks using a multilabel Informer neural network  
- 📊 **Interactive Analysis** — Drag-to-select wavelength regions and inspect detailed spectral features in real-time  
- 🔬 **Scientific Rigor** — Baseline correction (Asymmetric Least Squares), peak detection, and statistical validation  
- 💾 **Batch Processing** — Process entire folders of spectra with saved parameter presets  
- 🎨 **Publication-Ready Plots** — Export high-resolution figures (300 DPI) with customizable labels  
- ⚙️ **Cross-Platform** — Windows, macOS, and Linux support via Python and Qt
+ ### Key Features
+ 
+ 🔬 **Physics-Based Synthetic Spectral Library** — Training spectra generated via Saha–Boltzmann plasma theory for robust multi-element representation  
+ 🤖 **Informer Encoder Architecture** — 2-layer ProbSparse attention mechanism for efficient processing of 4096-channel high-resolution spectra  
+ 🎯 **Multi-Label Classification** — Simultaneous detection of 17 elements + background class from a single LIBS spectrum  
+ 🌿 **Experimental Case Study** — Qualitative analysis of Aceh traditional women's medicine samples  
+ 📊 **Reproducible Workflow** — Complete scripts for model training, evaluation, and inference with documented hyperparameters  
+ 💻 **Interactive GUI** — PySide6-based graphical interface for real-time spectral visualization and element identification
```

### Change 3: Repository Structure Section (Added)

```diff
- ## Project Structure
+ ## Repository Structure

- ```
- spectroscopic-analyzer/
+ ```
+ informer-libs-aceh/
+ ├── app/                            # Application source code
+ │   ├── main.py                     # GUI application entry point
+ │   ├── model.py                    # Informer model and utility functions
+ │   ├── processing.py               # Spectral data preprocessing
+ │   ├── core/
+ │   │   └── analysis.py             # Complete analysis pipeline
+ │   └── ui/                         # GUI components (PySide6)
+ │       ├── main_window.py          # Main application window
+ │       └── ...
+ ├── assets/                         # Model weights and reference data
+ │   ├── informer_multilabel_model.pth    # Pretrained model weights
+ │   ├── element-map-17.json              # Element-wavelength mapping
+ │   └── wavelengths_grid.json            # Target wavelength grid (4096 channels)
+ ├── data/                           # Experimental and synthetic data
+ │   ├── synthetic/                  # Training data (Saha–Boltzmann spectra)
+ │   └── experimental/               # Case study measurements
+ ├── models/                         # Saved checkpoints and model definitions
+ ├── notebooks/                      # Jupyter notebooks for analysis
+ ├── training/                       # Training scripts
+ ├── scripts/                        # Utility and inference scripts
```

### Change 4: Citation Section Replaced with "How to Cite" and "Contact"

```diff
- ## Citation & Publication
- 
- If you use Spectroscopic Analyzer in your research, please cite:
- 
- ```bibtex
- @software{spectroscopic_analyzer_2025,
-   author = {Birrulwaldi Nurdin},
-   title = {Spectroscopic Analyzer: AI-Powered LIBS Analysis Software},
-   year = {2025},
-   url = {https://github.com/yourusername/spectroscopic-analyzer},
-   note = {v1.0.0}
- }
- ```
+ ## How to Cite
+ 
+ If you use this code or data in your research, please cite the accompanying paper:
+ 
+ ### Human-Readable Citation
+ 
+ Walidain, B., Idris, N., Saddami, K., Yuzza, N., & Mitaphonna, R. (2025). Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine. *IOP Conference Series: Earth and Environmental Science*, AIC 2025. doi: to be assigned
+ 
+ ### BibTeX
+ 
+ ```bibtex
+ @inproceedings{Walidain2025,
+   title={Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine},
+   author={Walidain, Birrul and Idris, Nasrullah and Saddami, Khairun and Yuzza, Natasya and Mitaphonna, Rara},
+   booktitle={AIC 2025 -- Natural Life and Sciences track},
+   journal={IOP Conference Series: Earth and Environmental Science},
+   year={2025},
+   doi = {to be assigned},
+   note={in press}
+ }
+ ```
+ 
+ ---
+ 
+ ## Contact
+ 
+ **Corresponding Author:**
+ - **Name**: Nasrullah Idris
+ - **Email**: [nasrullah.idris@usk.ac.id](mailto:nasrullah.idris@usk.ac.id)
+ - **Affiliation**: Department of Physics, Faculty of Mathematics and Natural Sciences, Universitas Syiah Kuala, Banda Aceh 23111, Indonesia
+ 
+ **GitHub Maintainer:**
+ - **Name**: Birrul Walidain
+ - **Repository**: [github.com/birrulwaldain/informer-libs-aceh](https://github.com/birrulwaldain/informer-libs-aceh)
```

---

## File 2: CITATION.cff

```diff
  cff-version: 1.2.0
- title: Spectroscopic Analyzer
+ title: Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine
  type: software
  authors:
-   - family-names: Nurdin
-     given-names: Birrulwaldi
-     orcid: "https://orcid.org/0000-0000-0000-0000"
+   - family-names: Walidain
+     given-names: Birrul
+   - family-names: Idris
+     given-names: Nasrullah
+   - family-names: Saddami
+     given-names: Khairun
+   - family-names: Yuzza
+     given-names: Natasya
+   - family-names: Mitaphonna
+     given-names: Rara
  description: >-
-   AI-powered Laser-Induced Breakdown Spectroscopy (LIBS) analysis software
-   with interactive visualization and publication-ready export capabilities.
+   Implementation and experimental data for an Informer-based deep learning model
+   for qualitative multi-element analysis via Laser-Induced Breakdown Spectroscopy (LIBS).
+   The model is trained on physics-based synthetic spectra generated using the Saha–Boltzmann
+   equation and evaluated on Aceh traditional herbal medicine samples.
  keywords:
    - LIBS
+   - Laser-Induced Breakdown Spectroscopy
+   - Informer
    - Spectroscopy
    - Deep Learning
    - Element Detection
+   - Aceh Traditional Medicine
+   - Saha-Boltzmann
+   - ProbSparse Attention
  license: MIT
- repository-code: "https://github.com/yourusername/spectroscopic-analyzer"
+ repository-code: "https://github.com/birrulwaldain/informer-libs-aceh"
  version: 1.0.0
  date-released: 2025-11-29
  
  preferred-citation:
-   type: software
+   type: conference-paper
    authors:
-     - family-names: Nurdin
-       given-names: Birrulwaldi
-   title: "Spectroscopic Analyzer: AI-Powered LIBS Analysis Software"
+     - family-names: Walidain
+       given-names: Birrul
+     - family-names: Idris
+       given-names: Nasrullah
+     - family-names: Saddami
+       given-names: Khairun
+     - family-names: Yuzza
+       given-names: Natasya
+     - family-names: Mitaphonna
+       given-names: Rara
+   title: "Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine"
+   journal: "IOP Conference Series: Earth and Environmental Science"
+   conference:
+     name: "AIC 2025 – Natural Life and Sciences track"
    year: 2025
-   version: 1.0.0
-   repository-code: "https://github.com/yourusername/spectroscopic-analyzer"
-   url: "https://github.com/yourusername/spectroscopic-analyzer"
+   status: "in press"
+   doi: "to be assigned"
+   repository-code: "https://github.com/birrulwaldain/informer-libs-aceh"
+   url: "https://github.com/birrulwaldain/informer-libs-aceh"
```

---

## File 3: app/main.py

```diff
+ """
+ Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine
+ 
+ Interactive GUI application for spectroscopic data analysis using a deep learning model.
+ 
+ This implementation accompanies the paper:
+     Walidain, B., Idris, N., Saddami, K., Yuzza, N., & Mitaphonna, R. (2025).
+     "Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine."
+     IOP Conference Series: Earth and Environmental Science, AIC 2025. (in press)
+ 
+ Authors:
+     Birrul Walidain, Nasrullah Idris, Khairun Saddami, Natasya Yuzza, Rara Mitaphonna
+ 
+ For more information, see:
+     - README.md: Installation and usage instructions
+     - GitHub: https://github.com/birrulwaldain/informer-libs-aceh
+ """
  from __future__ import annotations
  import os
  import sys
```

---

## File 4: app/model.py

```diff
+ """
+ Neural network models for Informer-Based LIBS analysis.
+ 
+ This module implements the Informer architecture for multi-element classification
+ from laser-induced breakdown spectroscopy (LIBS) data.
+ 
+ Paper Reference:
+     Walidain, B., Idris, N., Saddami, K., Yuzza, N., & Mitaphonna, R. (2025).
+     "Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine."
+     IOP Conference Series: Earth and Environmental Science, AIC 2025. (in press)
+ 
+ Informer Architecture Reference:
+     Zhou, H., Zhang, S., Peng, J., et al. (2021).
+     "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting."
+     ICLR 2021. arXiv:2012.07436
+ 
+ See Also:
+     - GitHub: https://github.com/birrulwaldain/informer-libs-aceh
+     - README: Installation and usage instructions
+ """
  
  # app/model.py
```

---

## File 5: app/processing.py

```diff
+ """
+ Data preprocessing utilities for LIBS spectroscopic analysis.
+ 
+ Implements spectral preprocessing operations including baseline correction,
+ normalization, and data loading for the Informer-based LIBS analysis pipeline.
+ 
+ Paper Reference:
+     Walidain, B., Idris, N., Saddami, K., Yuzza, N., & Mitaphonna, R. (2025).
+     "Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine."
+     IOP Conference Series: Earth and Environmental Science, AIC 2025. (in press)
+ 
+ See Also:
+     - GitHub: https://github.com/birrulwaldain/informer-libs-aceh
+     - README: Installation and usage instructions
+ """
  
  # app/processing.py
```

---

## File 6: setup.py

```diff
  #!/usr/bin/env python
  """
- Setup configuration for Spectroscopic Analyzer package.
+ Setup configuration for Informer-Based LIBS analysis package.
+ 
+ This package accompanies the paper:
+     Walidain, B., Idris, N., Saddami, K., Yuzza, N., & Mitaphonna, R. (2025).
+     "Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine."
+     IOP Conference Series: Earth and Environmental Science, AIC 2025. (in press)
  """
  
- setup(
-     name="spectroscopic-analyzer",
+ setup(
+     name="informer-libs-aceh",
      version="1.0.0",
-     author="Birrulwaldi Nurdin",
-     author_email="birrulwaldi@example.com",
-     description="AI-powered LIBS spectroscopic analysis and element detection",
+     author="Birrul Walidain, Nasrullah Idris, Khairun Saddami, Natasya Yuzza, Rara Mitaphonna",
+     author_email="nasrullah.idris@usk.ac.id",
+     maintainer="Birrul Walidain",
+     description="Informer-based deep learning for qualitative multi-element LIBS analysis",
      long_description=long_description,
      long_description_content_type="text/markdown",
-     url="https://github.com/yourusername/spectroscopic-analyzer",
+     url="https://github.com/birrulwaldain/informer-libs-aceh",
      project_urls={
-         "Bug Tracker": "https://github.com/yourusername/spectroscopic-analyzer/issues",
-         "Documentation": "https://github.com/yourusername/spectroscopic-analyzer/docs",
-         "Source Code": "https://github.com/yourusername/spectroscopic-analyzer",
+         "Bug Tracker": "https://github.com/birrulwaldain/informer-libs-aceh/issues",
+         "Documentation": "https://github.com/birrulwaldain/informer-libs-aceh",
+         "Source Code": "https://github.com/birrulwaldain/informer-libs-aceh",
+         "Paper": "https://iopscience.iop.org/",
      },
      ...
      entry_points={
          "console_scripts": [
-             "spectroscopic-analyzer=app.main:main",
+             "informer-libs=app.main:main",
          ],
      },
      ...
      keywords=[
          "LIBS",
+         "Laser-Induced Breakdown Spectroscopy",
+         "Informer",
          "deep learning",
+         "multi-element analysis",
+         "Aceh traditional medicine",
+         "Saha-Boltzmann",
          "element detection",
          "scientific software",
      ],
```

---

## Summary of Changes

**Total files modified:** 6
- README.md: Title, overview, key features, repository structure, citation, and contact sections updated
- CITATION.cff: Complete author list, journal info, and conference details added
- app/main.py: Paper citation header added
- app/model.py: Paper citation header added
- app/processing.py: Paper citation header added
- setup.py: Package name, authors, description, and keywords updated

**No code logic changed:** All modifications are documentation and metadata only.

**Consistency verified:** All references to project name, authors, and repository URL are now consistent across all files.

