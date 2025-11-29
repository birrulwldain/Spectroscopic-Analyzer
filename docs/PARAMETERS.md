# Parameter Reference Guide

Detailed explanation of all analysis parameters and their effects.

---

## Preprocessing Parameters

### Baseline Correction (ALS)

#### `baseline_lambda` (λ)
- **Type**: Float
- **Default**: `1e5` (100,000)
- **Range**: 1e2 to 1e9
- **Effect**: Controls baseline smoothness

**Lower values** (1e2 - 1e4):
- More flexible baseline
- Follows spectrum contours closely
- May not fully remove broad background
- Use for: Spectra with structured background

**Higher values** (1e6 - 1e9):
- Very smooth baseline
- Treats background as uniform
- May overfit sharp features
- Use for: Clean spectra with flat background

**Recommendation**: Start with 1e5, adjust based on visual inspection.

```python
# Example: Reduce baseline to 1e4 for better feature preservation
preprocessed = prepare_asc_data(
    content,
    target_wavelengths,
    als_lambda=1e4  # Reduced from 1e5
)
```

---

#### `baseline_p` (p)
- **Type**: Float
- **Default**: `0.001`
- **Range**: 0.0001 to 0.1
- **Effect**: Baseline asymmetry (penalizes underestimation)

**Lower values** (0.0001 - 0.001):
- Baseline emphasizes valleys
- Better for spectra with baseline below peaks
- Use for: Normal LIBS spectra

**Higher values** (0.01 - 0.1):
- More symmetric baseline
- Less emphasis on valleys
- Use for: Baseline above some peaks

**Recommendation**: Keep at 0.001 for standard LIBS data.

---

#### `baseline_max_iter`
- **Type**: Integer
- **Default**: `10`
- **Range**: 1 to 50
- **Effect**: ALS algorithm iterations

**More iterations**:
- Better convergence
- Slower processing
- Marginal improvement after ~10 iterations

**Recommendation**: 10 is sufficient for most cases.

---

### Intensity Normalization

#### `target_max_intensity`
- **Type**: Float
- **Default**: `0.8`
- **Range**: 0.1 to 1.0
- **Effect**: Maximum intensity after normalization

**Lower values** (0.1 - 0.5):
- Leaves headroom in data range
- Can help with numerical stability
- Loses dynamic range

**Higher values** (0.8 - 1.0):
- Uses full data range
- Better utilizes model input space
- Recommended for deep learning models

**Recommendation**: 0.8 for good balance, 1.0 for ML models.

---

## Peak Detection Parameters

### `peak_height`
- **Type**: Float (0-1 normalized)
- **Default**: `0.01`
- **Range**: 0.001 to 0.5
- **Effect**: Minimum height for peak detection

**Lower values** (0.001 - 0.01):
- Detects weak peaks and noise
- May include false positives
- Use for: Weak element lines

**Higher values** (0.1 - 0.5):
- Detects only strong peaks
- Misses weak elements
- Use for: High SNR spectra

**Recommendation**: Start with 0.01, adjust if getting too many/few peaks.

---

### `peak_prominence`
- **Type**: Float (0-1 normalized)
- **Default**: `0.01`
- **Range**: 0.001 to 0.5
- **Effect**: Minimum prominence (height above baseline)

**How it works**:
```
Peak height
     ↑     /\
     |    /  \         ← Prominence = height above adjacent valleys
     |___/    \___
     ↓
Baseline
```

**Lower values** (0.001 - 0.01):
- Detects peaks close to baseline
- More false positives
- Use for: Weak spectra

**Higher values** (0.05 - 0.2):
- Only prominent peaks
- Fewer false positives
- Use for: Noisy spectra

**Recommendation**: 0.01 for standard, increase for noisy data.

---

### `peak_distance`
- **Type**: Integer (samples)
- **Default**: `10`
- **Range**: 1 to 100
- **Effect**: Minimum distance between peak centers

**Lower values** (1 - 5):
- Detects closely-spaced peaks (e.g., fine structure)
- May split single broad peaks
- More peaks detected

**Higher values** (20 - 100):
- Only well-separated peaks
- Merges close peaks
- Fewer peaks detected

**Interpretation**: At 18000 sample wavelength grid (250-800nm):
- 1 sample ≈ 0.03 nm
- 10 samples ≈ 0.3 nm
- 100 samples ≈ 3 nm

**Recommendation**: 10-20 for standard resolution.

---

### `peak_max`
- **Type**: Integer
- **Default**: `50`
- **Range**: 1 to 500
- **Effect**: Maximum number of peaks to detect

**Lower values** (10 - 30):
- Only strongest peaks
- Misses weak elements
- Faster processing

**Higher values** (50 - 200):
- Detects all peaks above threshold
- More false positives
- Slower processing

**Recommendation**: 50-100 for complete analysis, 20 for high-SNR spectra.

---

## Element Detection Parameters

### `confidence_threshold`
- **Type**: Float (probability)
- **Default**: `0.5`
- **Range**: 0.0 to 1.0
- **Effect**: Minimum confidence for element detection

**How it works**: Only elements with model probability > threshold are reported.

**Lower values** (0.1 - 0.3):
- More detected elements (high recall)
- More false positives
- Use for: Exploratory analysis

**Higher values** (0.5 - 0.9):
- Fewer detected elements (high precision)
- Fewer false positives
- Use for: Publication/validation

**Recommendation**:
- 0.3: Exploratory (find all possible elements)
- 0.5: Balanced (default)
- 0.7+: Conservative (high confidence only)

```python
# Example: Only report highly confident predictions
result = run_full_analysis({
    ...
    'confidence_threshold': 0.8
})
```

---

## Advanced Parameters

### Abel Deconvolution

#### `enable_abel`
- **Type**: Boolean
- **Default**: `False`
- **Effect**: Enables radial profile reconstruction

When enabled:
- Assumes cylindrical symmetry
- Applies inverse Abel transform
- Generates radial intensity profile
- Requires additional computation (~1-2s)

**Use when**: Sample has cylindrical geometry (plasma column, fiber, etc.)

```python
result = run_full_analysis({
    ...
    'enable_abel': True
})
# result['abel_profile'] contains radial profile
```

---

### Validation Parameters

#### `peak_width`
- **Type**: Float
- **Default**: `None` (auto)
- **Effect**: Expected peak width in samples

When specified:
- Improves peak detection accuracy
- Filters peaks by width
- Useful for known line widths

---

## Parameter Presets

Pre-configured parameter sets for common scenarios:

### Preset: High-SNR Spectrum
```python
preset_high_snr = {
    'baseline_lambda': 1e5,
    'baseline_p': 0.001,
    'peak_height': 0.05,        # Higher threshold
    'peak_prominence': 0.05,
    'peak_distance': 20,         # Wider separation
    'peak_max': 30,              # Fewer peaks
    'confidence_threshold': 0.7,
    'enable_abel': False,
}
```

**Use for**: Clean spectra, industrial applications

---

### Preset: Low-SNR Spectrum
```python
preset_low_snr = {
    'baseline_lambda': 1e4,      # Smoother baseline
    'baseline_p': 0.005,
    'peak_height': 0.005,        # Lower threshold
    'peak_prominence': 0.005,
    'peak_distance': 5,          # Tighter spacing
    'peak_max': 100,             # More peaks
    'confidence_threshold': 0.3, # Lower confidence
    'enable_abel': False,
}
```

**Use for**: Weak signal, exploratory analysis

---

### Preset: Quantitative Analysis
```python
preset_quantitative = {
    'baseline_lambda': 5e4,      # Intermediate
    'baseline_p': 0.001,
    'peak_height': 0.02,
    'peak_prominence': 0.02,
    'peak_distance': 15,
    'peak_max': 50,
    'confidence_threshold': 0.6, # Higher confidence
    'enable_abel': True,         # For radial info
}
```

**Use for**: Concentration calibration, publication

---

## Workflow: Parameter Optimization

### Step 1: Load Spectrum
```
Load your sample spectrum
Look at the raw data
Note: Signal level, noise, background
```

### Step 2: Adjust Baseline
```
Try: lambda = 1e5 (default)
Look at: Preprocessed plot
Question: Does baseline follow broad features?
  - If too wiggly: increase lambda (1e6)
  - If too smooth: decrease lambda (1e4)
```

### Step 3: Fine-Tune Peak Detection
```
Look at: Detected peaks in detail plot
Question: Getting too many peaks?
  - If yes: increase peak_height or peak_prominence
  - Try: 0.01 → 0.02 → 0.05
Question: Missing weak peaks?
  - If yes: decrease peak_height
  - Try: 0.01 → 0.005 → 0.001
```

### Step 4: Set Confidence Threshold
```
Look at: Detected elements and their probabilities
Question: Too many false positives?
  - Increase confidence_threshold (0.5 → 0.7)
Question: Missing known elements?
  - Decrease confidence_threshold (0.5 → 0.3)
```

### Step 5: Validate & Export
```
Compare detected elements with expected
Visual inspection of peaks and labels
Export high-quality figure
Save parameter preset for future use
```

---

## Scientific References

### Baseline Correction
- **Eilers, P. H., & Boelens, H. F.** (2005). "Baseline correction with asymmetric least squares smoothing." Technical report, Leiden University.
- Algorithm: Iteratively reweighted least squares (IRLS)
- Paper demonstrates on chromatography data (same principle applies)

### Peak Detection
- **Virtanen, P., et al.** (2020). "SciPy 1.0: fundamental algorithms for scientific computing." Nature Methods, 17(3), 261-272.
- Implementation: scipy.signal.find_peaks()
- References: Signal processing literature

### LIBS Analysis
- **Cremers, D. A., & Radziemski, L. M.** (2006). Handbook of Laser-Induced Breakdown Spectroscopy: Physics, Diagnostics, and Applications. John Wiley & Sons.
- Comprehensive LIBS reference

---

**Last Updated**: November 2025

