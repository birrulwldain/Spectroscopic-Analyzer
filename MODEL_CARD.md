# Model Card: Informer-Based LIBS Multi-Element Classifier

## Overview

This document provides technical details about the Informer-based deep learning model for multi-element qualitative analysis via Laser-Induced Breakdown Spectroscopy (LIBS).

**Model Name:** `informer_multilabel_model.pth`  
**Version:** 1.0.0  
**Release Date:** November 2025  
**Paper:** "Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine" (IOP EES, AIC 2025)

---

## Model Architecture

### Overview

A specialized Informer encoder designed for high-dimensional spectroscopic data classification.

### Architecture Details

| Component | Configuration |
|-----------|---|
| **Input Shape** | (batch, 4096) → (batch, 4096, 1) |
| **Encoder Layers** | 2 layers |
| **Embedding Dimension (d_model)** | 32 |
| **Feed-Forward Dimension (d_ff)** | 64 |
| **Number of Attention Heads** | 4 |
| **Attention Mechanism** | ProbSparse (Informer 2020) |
| **Dropout Rate** | 0.1 |
| **Activation** | GELU |
| **Output Dimension** | 18 (17 elements + background) |
| **Output Activation** | Sigmoid (multi-label) |

### Model Flow

```
Input Spectrum (4096 channels)
         ↓
  Embedding Layer (→ d_model=32)
         ↓
  ┌─────────────────────┐
  │ Informer Encoder    │
  │ Layer 1             │
  │ (ProbSparse Attn)   │
  └─────────────────────┘
         ↓
  ┌─────────────────────┐
  │ Informer Encoder    │
  │ Layer 2             │
  │ (ProbSparse Attn)   │
  └─────────────────────┘
         ↓
  Global Average Pooling
         ↓
  Classification Head
  (Linear → 18)
         ↓
  Sigmoid Activation
         ↓
  Output Probabilities (0-1 per element)
```

### Why Informer for LIBS?

- **Efficiency:** ProbSparse attention reduces O(L²) to O(L log L) complexity, enabling processing of 4096-channel spectra
- **Long-Range Dependencies:** Captures broad absorption/emission patterns across wavelength ranges
- **Physics Alignment:** Attention patterns can reflect physical transitions and element-specific spectral features
- **Proven Track Record:** Transformer architecture established for sequential/time-series data

---

## Training Data

### Synthetic Data Generation

**Method:** Physics-based Saha–Boltzmann equation  
**Tool:** Custom Python implementation (scripts/job.py, scripts/planner.py)

### Data Composition

| Aspect | Details |
|--------|---------|
| **Training Samples** | 8,000 synthetic spectra |
| **Validation Samples** | 1,000 synthetic spectra |
| **Test Samples** | 1,000 synthetic spectra |
| **Wavelength Range** | 200–850 nm |
| **Spectral Resolution** | ~0.16 nm/channel (4096 channels) |
| **Elements Detected** | 17 elements + background (Ca, C, H, K, Mg, Mn, N, Na, O, P, S, Si, Al, Fe, Cu, Zn, B, and background) |
| **Noise Levels** | 0.01, 0.02, 0.05 (1%, 2%, 5% Gaussian noise) |
| **Temperature Range** | 5000–15000 K |
| **Pressure Range** | 0.5–2.0 atm |

### Physics Model

Spectral intensity is calculated using:

$$I_\lambda = n_i \sigma_{ij}(\lambda) \exp\left(-\frac{E_j}{k_B T}\right) / Z(T)$$

where:
- $n_i$ = ion population (Saha–Boltzmann)
- $\sigma_{ij}(\lambda)$ = transition cross-section
- $T$ = plasma temperature
- $Z(T)$ = partition function

### Data Splits

- **Training:** 8,000 samples (80% of total)
- **Validation:** 1,000 samples (10% of total)
- **Test:** 1,000 samples (10% of total)

All splits contain balanced element representations and noise levels.

---

## Training Details

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Standard for deep learning; weight decay prevents overfitting |
| Learning Rate | 0.001 | Balances convergence speed and stability |
| Weight Decay | 1e-5 | L2 regularization for model regularization |
| Batch Size | 32 | Balance between memory and gradient quality |
| Dropout | 0.1 | Mild regularization to prevent overfitting |
| Epochs | 100 | With early stopping on validation loss |
| Loss Function | Focal Loss (multi-label) | Handles class imbalance in multi-label setting |
| Class Weights | Balanced | Inverse frequency weighting for rare elements |

### Training Procedure

1. **Data Preprocessing:**
   - Normalize spectra to [0, 1] range
   - Apply Asymmetric Least Squares (ALS) baseline correction
   - Clip outliers at 5σ

2. **Augmentation:**
   - Random Gaussian noise (σ = 0.01–0.05)
   - Wavelength jitter (±0.5 nm)
   - Intensity scaling (0.95–1.05×)

3. **Optimization:**
   - Warmup for 5 epochs (linear scheduler)
   - Cosine annealing for remaining 95 epochs
   - Gradient clipping at norm = 1.0

4. **Monitoring:**
   - Validation every epoch
   - Early stopping if val_loss plateaus for 15 epochs
   - Checkpoint best model by Hamming loss

### Training Hardware

| Setting | Spec |
|---------|------|
| **Tested on** | NVIDIA RTX 3080 (10 GB VRAM) |
| **Framework** | PyTorch 2.0+ |
| **Precision** | Float32 |
| **Estimated Time** | ~8 minutes (GPU), ~45 minutes (CPU) |

---

## Model Performance

### Overall Metrics (Test Set)

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.899 ± 0.015 |
| **Hamming Loss** | 0.087 ± 0.012 |
| **Macro F1-Score** | 0.878 ± 0.018 |
| **Micro F1-Score** | 0.891 ± 0.014 |
| **Weighted F1-Score** | 0.890 ± 0.015 |

### Per-Element Metrics (Table: perf_detail)

Example per-element performance from test set:

| Element | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| H | 0.92 ± 0.03 | 0.88 ± 0.04 | 0.90 ± 0.03 | 145 |
| C | 0.89 ± 0.03 | 0.86 ± 0.04 | 0.87 ± 0.03 | 128 |
| N | 0.85 ± 0.04 | 0.82 ± 0.05 | 0.83 ± 0.04 | 110 |
| O | 0.90 ± 0.03 | 0.91 ± 0.02 | 0.90 ± 0.03 | 155 |
| Ca | 0.88 ± 0.03 | 0.87 ± 0.04 | 0.87 ± 0.03 | 132 |
| ... | ... | ... | ... | ... |

(See paper Table `perf_detail` for complete list)

### Confusion & Performance

- **Best Detection:** H, O, Ca (F1 > 0.88)
- **Challenging Elements:** Rare/weak emitters (F1 ~ 0.81)
- **False Positive Rate:** ~2% (element detected when absent)
- **False Negative Rate:** ~8% (element missed when present)

---

## Inference Details

### Input Format

- **Type:** 1D NumPy array or PyTorch tensor
- **Shape:** (4096,) → spectrum over 200–850 nm at 0.16 nm resolution
- **Range:** [0, 1] (normalized intensity)
- **Preprocessing:** ALS baseline correction, min-max normalization

### Output Format

- **Type:** NumPy array or tensor
- **Shape:** (18,) → probability per element + background
- **Range:** [0, 1] per element (sigmoid output)
- **Interpretation:** P(element_i | spectrum) ≈ output[i]

### Classification Threshold

- **Default:** 0.5
- **Recommendation:** Adjust based on application
  - High precision (minimize false positives): 0.6–0.7
  - High recall (minimize false negatives): 0.3–0.4
  - Balanced (default): 0.5

### Inference Time

- **Single Spectrum:** ~1 ms (GPU), ~5 ms (CPU)
- **Batch (32 spectra):** ~15 ms (GPU), ~100 ms (CPU)

---

## Intended Use & Applications

### Primary Use Cases

1. **Qualitative Element Detection:** Identify which elements are present in a sample
2. **Rapid Sample Screening:** Fast multi-element analysis without quantification
3. **Material Classification:** Distinguish sample types based on elemental composition
4. **Research & Development:** Support in herbal medicine, minerals, and alloys

### Intended Users

- Spectroscopy researchers and technicians
- Materials scientists
- Environmental monitoring specialists
- Educational/training institutions

### Potential Applications

- Archaeological artifact analysis
- Herbal medicine validation (primary case study)
- Soil and sediment analysis
- Quality control in manufacturing

---

## Limitations & Biases

### Model Limitations

1. **Synthetic Training Data Only**
   - Model trained on simulated spectra; real experimental data may differ
   - Plasma conditions in simulations (LTE assumption) may not hold universally
   - Noise model (Gaussian) may not match real instrumental noise

2. **No Quantification**
   - Outputs element presence/absence probabilities, not concentrations
   - Cannot determine stoichiometry or abundance ratios

3. **Fixed Wavelength Grid**
   - Designed for 200–850 nm range at 0.16 nm/channel
   - Requires interpolation/resampling for other spectral resolutions

4. **Limited Element Set**
   - Trained on 17 elements + background
   - Cannot detect unlisted elements
   - Adding new elements requires retraining

5. **Multi-Label Assumption**
   - Model assumes independent element probabilities
   - Does not model element correlations or physical constraints

### Known Biases

1. **Element Imbalance:** Common elements (H, O, C) may have better performance than rare elements
2. **Noise Sensitivity:** Performance degrades gracefully with high noise (>5%)
3. **Temperature Dependency:** Trained for 5000–15000 K; performance outside range uncertain

### Handling Limitations

- **For quantification:** Post-process with chemometrics (e.g., PLSR)
- **For new elements:** Fine-tune model on labeled experimental data
- **For robustness:** Ensemble predictions or Bayesian uncertainty estimation

---

## Data & Model Provenance

### Model Card Version

| Item | Value |
|------|-------|
| **Card Version** | 1.0.0 |
| **Model Version** | 1.0.0 |
| **Last Updated** | November 30, 2025 |
| **Created By** | Birrul Walidain, Nasrullah Idris, et al. |

### Data Availability

- **Synthetic Training Data:** Generated on-demand via `scripts/job.py`
- **Experimental Data:** Available upon request (privacy concerns)
- **Model Weights:** `assets/informer_multilabel_model.pth`
- **Reference Data:** `assets/element-map-17.json`, `assets/wavelengths_grid.json`

### Citation

If you use this model, please cite:

```bibtex
@inproceedings{Walidain2025,
  title={Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine},
  author={Walidain, Birrul and Idris, Nasrullah and Saddami, Khairun and Yuzza, Natasya and Mitaphonna, Rara},
  booktitle={AIC 2025 -- Natural Life and Sciences track},
  journal={IOP Conference Series: Earth and Environmental Science},
  year={2025},
  note={In Press}
}
```

---

## Recommendations for Users

### Before Deployment

- [ ] Test on your specific LIBS instrument and samples
- [ ] Validate against reference materials with known composition
- [ ] Check preprocessing pipeline (baseline correction, normalization)
- [ ] Benchmark inference time on your hardware
- [ ] Consider model uncertainty via ensemble or dropout-based estimates

### For Best Results

- [ ] Preprocess all input spectra identically (ALS baseline, normalization)
- [ ] Use consistent integration time and settings on your LIBS system
- [ ] Calibrate wavelength grid (check 200–850 nm alignment)
- [ ] Filter outliers or corrupted spectra before inference
- [ ] Document any instrument-specific adjustments

### For Improvement

- [ ] Collect real experimental spectra and fine-tune model
- [ ] Add domain knowledge constraints (e.g., physical relationships)
- [ ] Implement active learning for targeted sample acquisition
- [ ] Explore quantitative extensions (Gaussian process regression)

---

## Contact & Support

**Model Developer:**  
Birrul Walidain  
Email: birrul@mhs.usk.ac.id  
GitHub: [github.com/birrulwaldain](https://github.com/birrulwaldain)

**Corresponding Author (Paper):**  
Nasrullah Idris  
Email: nasrullah.idris@usk.ac.id  
Universitas Syiah Kuala, Indonesia

**For Issues, Questions, or Improvements:**  
Open an issue at [github.com/birrulwaldain/informer-libs-multielement/issues](https://github.com/birrulwaldain/informer-libs-multielement/issues)

---

## Changelog

### v1.0.0 (November 2025)
- Initial release accompanying IOP EES paper
- Informer encoder (2 layers, d_model=32, n_heads=4)
- Multi-label focal loss training
- Per-element F1 ~0.87 ± 0.04 (weighted average)

---

**Status:** Production Ready  
**License:** MIT  
**Last Verified:** November 30, 2025

