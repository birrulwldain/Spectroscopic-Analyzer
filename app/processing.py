"""
Data preprocessing utilities for LIBS spectroscopic analysis.

Implements spectral preprocessing operations including baseline correction,
normalization, and data loading for the Informer-based LIBS analysis pipeline.

Paper Reference:
    Walidain, B., Idris, N., Saddami, K., Yuzza, N., & Mitaphonna, R. (2025).
    "Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine."
    IOP Conference Series: Earth and Environmental Science, AIC 2025. (in press)

See Also:
    - GitHub: https://github.com/birrulwaldain/informer-libs-aceh
    - README: Installation and usage instructions
"""

# app/processing.py

import pandas as pd
import numpy as np
from io import StringIO
from scipy import sparse
from scipy.sparse.linalg import spsolve

def als_baseline(y, lam, p, niter=10):
    """Asymmetric Least Squares smoothing for baseline correction."""
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    z = np.zeros(L)  # Initialize z
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def prepare_asc_data(
    asc_content_string: str,
    target_wavelengths: np.ndarray,
    target_max_intensity: float = 0.8,
    als_lambda: float = 1e6,
    als_p: float = 0.01,
    als_max_iter: int = 10
) -> np.ndarray:
    """Memproses string konten file .asc menjadi spektrum yang siap diprediksi menggunakan ALS untuk baseline correction."""
    try:
        df = pd.read_csv(StringIO(asc_content_string), sep=r'\s+', names=['wavelength', 'intensity'], comment='#')
        original_wavelengths, original_spectrum = df['wavelength'].values, df['intensity'].values
    except Exception as e:
        raise ValueError(f"Gagal mem-parsing data ASC. Pastikan formatnya benar. Detail: {e}")

    # Koreksi Baseline menggunakan ALS
    baseline = als_baseline(original_spectrum, als_lambda, als_p, als_max_iter)
    spectrum_corrected = original_spectrum - baseline
    spectrum_corrected[spectrum_corrected < 0] = 0

    # Normalisasi
    max_val = np.max(spectrum_corrected)
    processed_spectrum = (spectrum_corrected / max_val) * target_max_intensity if max_val > 0 else spectrum_corrected

    # Resampling
    resampled_spectrum = np.interp(target_wavelengths, original_wavelengths, processed_spectrum)
    return resampled_spectrum.astype(np.float32)