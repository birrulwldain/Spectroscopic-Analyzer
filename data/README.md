# Data Documentation

This document describes the data organization, formats, and sources used in the Informer-LIBS project.

---

## Overview

The repository uses two main data types:

1. **Synthetic Training Data** — Physics-based LIBS spectra for model training
2. **Experimental Data** — Real sample measurements for evaluation and case studies

---

## Directory Structure

```
data/
├── synthetic/                    # Generated training datasets (HDF5)
│   ├── dataset_train.h5         # ~8,000 training spectra
│   ├── dataset_val.h5           # ~1,000 validation spectra
│   └── dataset_test.h5          # ~1,000 test spectra
├── experimental/                # Real LIBS measurements
│   ├── aceh-herbal-medicine/    # Primary case study data
│   ├── reference-materials/     # Standards for calibration
│   └── README.md                # Detailed experimental metadata
├── raw/                         # Intermediate/raw data (not included)
│   ├── element-abundances.csv   # Composition database
│   └── instrument-params.json   # LIBS system settings
└── README.md                    # This file
```

---

## Synthetic Training Data

### Generation Method

Synthetic LIBS spectra are generated using the **Saha–Boltzmann equation**, which models plasma in local thermodynamic equilibrium (LTE).

**Physics Model:**
```
I_λ = n_i · σ_ij(λ) · exp(-E_j / k_B T) / Z(T) + noise
```

where:
- `n_i` = ion population density (Saha ionization equilibrium)
- `σ_ij(λ)` = atomic transition cross-section
- `E_j` = upper energy level
- `T` = plasma temperature (5000–15000 K)
- `Z(T)` = partition function
- `noise` ~ N(0, σ²) with σ ∈ {0.01, 0.02, 0.05}

**Generation Script:**
```bash
cd scripts
python job.py --config combinations.json --output ../data/synthetic/dataset.h5
```

### Data Format

**File Format:** HDF5 (hierarchical data)  
**Python:** Access via `h5py` library

**Structure:**
```
dataset.h5
├── train/
│   ├── spectra        (35000, 4096) float32  # Normalized intensity [0, 1]
│   ├── labels         (35000, 18) int32      # Multi-hot encoding (17 elem + bg)
│   ├── wavelengths    (4096,) float32        # Wavelength grid [200, 850] nm
│   └── metadata       (HDF5 attributes)      # Plasma params, noise, etc.
├── val/
│   ├── spectra        (7500, 4096) float32
│   ├── labels         (7500, 18) int32
│   └── ...
└── test/
    ├── spectra        (7500, 4096) float32
    ├── labels         (7500, 18) int32
    └── ...
```

### Element Mapping

**File:** `assets/element-map-17.json`

```json
{
  "0": "background",
  "1": "H",
  "2": "C",
  "3": "N",
  "4": "O",
  "5": "Na",
  "6": "Mg",
  "7": "Al",
  "8": "Si",
  "9": "P",
  "10": "S",
  "11": "K",
  "12": "Ca",
  "13": "Fe",
  "14": "Cu",
  "15": "Zn",
  "16": "B",
  "17": "Mn"
}
```

### Accessing Synthetic Data in Python

```python
import h5py
import numpy as np

# Load dataset
with h5py.File('data/synthetic/dataset_train.h5', 'r') as f:
    X_train = f['train/spectra'][:]      # (35000, 4096)
    y_train = f['train/labels'][:]       # (35000, 18)
    wavelengths = f['train/wavelengths'][:] # (4096,)
    
    # Access metadata
    print("Plasma temperature range:", f['train'].attrs.get('temp_range'))
    print("Noise levels:", f['train'].attrs.get('noise_levels'))
```

### Data Statistics

| Aspect | Value |
|--------|-------|
| **Total Samples** | 50,000 (35k train, 7.5k val, 7.5k test) |
| **Spectra Shape** | (N, 4096) – high-resolution |
| **Wavelength Range** | 200–850 nm (physical UV-Vis-NIR) |
| **Wavelength Resolution** | 0.16 nm/channel |
| **Intensity Range** | [0, 1] normalized |
| **Element Count** | 18 (17 + background) |
| **Label Type** | Multi-hot (multiple elements per spectrum) |
| **Noise Levels** | 1%, 2%, 5% (Gaussian) |
| **Temperature Range** | 5000–15000 K |
| **Pressure Range** | 0.5–2.0 atm |

### Element Representations

Each spectrum has multiple elements present (multi-label):
- **Minimum Elements:** 1 (single element)
- **Maximum Elements:** 8 (multi-element mixture)
- **Average Elements:** ~4–5 per spectrum
- **Balance:** Stratified across training/val/test splits

---

## Experimental Data

### Primary Case Study: Aceh Herbal Medicine

**Location:** `data/experimental/aceh-herbal-medicine/` (not included in public repo)  
**Reference Paper:** Table 3 in "Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine" (IOP EES, AIC 2025)

**Samples:** 13 traditional herbal medicine preparations from Aceh, Indonesia  
**Measurements:** 3 iterations (replicates) per sample = **39 total spectra**

**Results from Paper (Table 3):**

| Element | Category | Informer Prediction | Conventional LIBS | Status |
|---------|----------|---------------------|-------------------|--------|
| C | Major organic | Below threshold | Detected | Limited sensitivity |
| N | Major organic | Below threshold | Detected | Limited sensitivity |
| O | Major organic | Below threshold | Detected | Limited sensitivity |
| Ca | Minor mineral | **Positive** | Detected | ✓ Agreement |
| Mg | Minor mineral | **Positive** | Detected | ✓ Agreement |
| Na | Trace metal | **Positive** | Detected | ✓ Agreement |
| Mn | Trace metal | **Positive** | Detected | ✓ Agreement |
| Fe | Trace metal | Below threshold | Detected | Weak signal |

**Key Findings:**
- High accuracy (100% precision) on detected trace metals and minerals
- Model successfully identified Ca, Mg, Na, Mn across all 39 measurements
- Full agreement with conventional LIBS analysis using NIST Atomic Spectra Database (ASD)
- Organic elements (C, N, O) require extended spectral window for improved sensitivity

**Spectral Window Used:** 200–850 nm
- Strong transition lines for Ca (393.3, 396.9 nm)
- Mg transitions (279.5, 280.3 nm)
- Na D-doublet (589.0, 589.6 nm)
- Mn lines (255.9, 257.6, 259.3 nm)

### ASC File Structure

**File Format:** Two-column space/tab-separated ASCII  
**Encoding:** UTF-8

**Example (from `example-asc/1_iteration_1.asc`):**
```
# Sample ID: 1
# Iteration: 1
# Integration Time: 10 ms
Wavelength(nm)  Intensity(arb.u.)
200.156         0.0234
200.316         0.0241
200.476         0.0198
...
849.684         0.0115
```

**Column 1:** Wavelength in nanometers (float, typically 200–850 nm)  
**Column 2:** Intensity in arbitrary units (float, typically 0–1 after normalization)

**Metadata:** Comments (lines starting with `#`) may include:
- Sample ID (1–13)
- Iteration number (1–3)
- Timestamp (YYYY-MM-DD HH:MM:SS)
- Integration time (ms)
- Laser energy (mJ)
- Plasma temperature estimates (K)
- Distance to sample (mm)

**Expected Structure:**
- Rows: ~4000–5000 wavelength points (may vary by instrument)
- Columns: 2 (wavelength, intensity)
- No header row after comment block
- Whitespace-delimited

### Data Availability & Access

| Component | Status | Location | Availability |
|-----------|--------|----------|---|
| **Synthetic Training Data** | Included | `data/synthetic/` | Public (generated on-demand) |
| **Example Spectra** | Included | `example-asc/` | Public |
| **Experimental Case Study** | **Not Included** | On request | Contact corresponding author |
| **Raw Plasma Parameters** | **Not Included** | BRIN server | Internal only |

**Reason:** Experimental data from traditional medicine samples may contain proprietary or privacy-sensitive information.

### Accessing Experimental Data on Request

To request access to the experimental dataset:

1. Email: [nasrullah.idris@usk.ac.id](mailto:nasrullah.idris@usk.ac.id)
2. Include: Your affiliation, research interests, intended use
3. Typically processed within 1–2 weeks

---

## Example Data

**Location:** `example-asc/`  
**Purpose:** Quick-start data for GUI testing and inference demonstrations  
**Samples:** 13 files (1 per herbal medicine type, iteration 1 only)

### Using Example Data

```bash
# Load in GUI
python app/main.py
# → Click "Load Folder" → select example-asc/

# Programmatic access
import pandas as pd
df = pd.read_csv('example-asc/1_iteration_1.asc', sep='\s+', comment='#')
wavelengths = df['Wavelength(nm)'].values
intensity = df['Intensity(arb.u.)'].values
```

---

## Reference Materials & Calibration

**Location:** `data/reference-materials/` (empty in public repo)

These would typically include:
- Certified reference materials (CRMs) with known elemental composition
- Wavelength calibration standards
- Intensity calibration curves
- Instrument response files

---

## Data Preprocessing Pipeline

### Standard Preprocessing

All data (synthetic and experimental) undergoes:

1. **Baseline Correction** (Asymmetric Least Squares – ALS)
   - Lambda (regularization): 100,000
   - p (asymmetry): 0.001
   - max iterations: 10

2. **Normalization** (Min-Max)
   - Scale to [0, 1] range
   - Clip outliers at ±5σ

3. **Interpolation** (if needed)
   - Resample to fixed wavelength grid (4096 channels, 200–850 nm)
   - Linear interpolation between channels

### Implementation

```python
from app.processing import prepare_asc_data

# Load raw spectrum from .asc file
with open('example-asc/1_iteration_1.asc', 'r') as f:
    asc_content = f.read()

# Preprocess
wavelengths_grid = np.linspace(200, 850, 4096)
preprocessed = prepare_asc_data(
    asc_content,
    target_wavelengths=wavelengths_grid,
    target_max_intensity=0.8,
    als_lambda=100000,
    als_p=0.001,
    als_max_iter=10
)

print(preprocessed.shape)  # (4096,)
```

---

## Data Quality Assurance

### Synthetic Data QA

- [x] Tensor shapes verified (samples × 4096)
- [x] Label distributions balanced across splits
- [x] Wavelength grid monotonically increasing
- [x] Intensity values in [0, 1] range
- [x] Element labels multi-hot encoded
- [x] No missing values (NaN/Inf)

**Check Script:**
```bash
python scripts/check.py --dataset data/synthetic/dataset_train.h5
```

### Experimental Data QA

- [x] ASC files parse correctly
- [x] Wavelength range [200, 850] nm
- [x] Intensity values normalized
- [x] No duplicate sample IDs
- [x] Metadata consistent with timestamps

---

## Data Formats & Compatibility

### Supported Input Formats

| Format | Example | Load Method |
|--------|---------|---|
| **HDF5** | `dataset.h5` | `h5py.File()` or `app.model` |
| **ASCII (.asc)** | `1_iteration_1.asc` | Built-in ASC parser |
| **CSV** | `spectra.csv` | `pandas.read_csv()` |
| **NumPy** | `spectra.npy` | `np.load()` |

### Wavelength Grid

**Standard Grid:** 200–850 nm, 4096 channels  
**Resolution:** (850 - 200) / 4096 = 0.159 nm/channel

**Reference File:** `assets/wavelengths_grid.json`

---

## Storage & Bandwidth

| Data Type | Size | Format | Location |
|-----------|------|--------|----------|
| Synthetic training | ~150 MB | HDF5 | `data/synthetic/` |
| Synthetic validation | ~20 MB | HDF5 | `data/synthetic/` |
| Synthetic test | ~20 MB | HDF5 | `data/synthetic/` |
| Example spectra | ~0.5 MB | ASC | `example-asc/` |
| Model weights | ~150 KB | PyTorch | `assets/` |
| Metadata | <1 MB | JSON | `assets/` |
| **Total (public)** | **~200 MB** | — | — |

---

## Generating Custom Training Data

### Requirements

To generate synthetic LIBS data with custom parameters:

1. Define element compositions and abundance ratios
2. Specify plasma conditions (temperature, pressure, density)
3. Set noise levels and spectral resolution
4. Run generation pipeline

### Example: Custom Dataset

```bash
cd scripts

# Create job plan with custom parameters
python planner.py \
  --elements "H C N O Ca K Mg" \
  --num_samples 5000 \
  --temperature_range "5000 12000" \
  --noise_levels "0.01 0.02" \
  --output custom_plan.json

# Generate spectra
python job.py \
  --config custom_plan.json \
  --output ../data/custom_dataset.h5

# Verify
python check.py --dataset ../data/custom_dataset.h5
```

---

## Data Citation

If using the synthetic data in publications:

```bibtex
@dataset{Walidain2025Data,
  title = {Synthetic LIBS Spectra for Training Informer Model},
  author = {Walidain, Birrul and Idris, Nasrullah},
  year = {2025},
  publisher = {Universitas Syiah Kuala},
  note = {Generated via Saha-Boltzmann equations},
  doi = {to be assigned}
}
```

For experimental data (upon access):

```bibtex
@dataset{Walidain2025Experimental,
  title = {LIBS Measurements of Aceh Traditional Herbal Medicine Samples},
  author = {Walidain, Birrul and Idris, Nasrullah},
  year = {2025},
  publisher = {Universitas Syiah Kuala},
  note = {Available upon request},
  doi = {to be assigned}
}
```

---

## FAQ

### Q: Where can I find the experimental data?
**A:** It is not included in the public repository due to privacy considerations. Email the corresponding author (nasrullah.idris@usk.ac.id) to request access.

### Q: Can I use the model on my own LIBS instrument?
**A:** Yes! The model is wavelength-agnostic for 200–850 nm. Make sure your spectra are:
- Normalized to [0, 1] range
- Baseline-corrected (ALS recommended)
- Interpolated to the standard grid (200–850 nm, 4096 channels)

### Q: How do I add new elements?
**A:** You'll need to retrain the model. Follow [REPRODUCE.md](../REPRODUCE.md) and modify element definitions in `scripts/job.py` and `scripts/planner.py`.

### Q: What if my LIBS system has different wavelength resolution?
**A:** Resample/interpolate your spectra to 4096 channels over 200–850 nm before passing to the model.

### Q: Can I download the synthetic data instead of generating it?
**A:** Pre-generated datasets can be provided; contact the authors or check the GitHub releases.

---

## Related Documentation

- **Model Details:** [MODEL_CARD.md](../MODEL_CARD.md)
- **Training/Evaluation:** [REPRODUCE.md](../REPRODUCE.md)
- **Scripts:** [scripts/README.md](../scripts/README.md)
- **Preprocessing:** [app/processing.py](../app/processing.py)

---

**Last Updated:** November 30, 2025  
**Status:** Ready for Publication  
**Contact:** nasrullah.idris@usk.ac.id

