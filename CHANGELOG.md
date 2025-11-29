# Changelog

All notable changes to Spectroscopic Analyzer are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Batch processing dialog for analyzing multiple files
- Parameter preset system for saving/loading analysis configurations
- NIST database integration for reference spectral lines
- Peak fitting and deconvolution capabilities
- Quantitative analysis mode with calibration support
- PDF report generation
- Unit test suite

### Changed
- Improved GUI layout with better panel organization
- Enhanced error messages and logging
- Optimized baseline correction algorithm
- Updated documentation

### Fixed
- File export path validation
- Memory leaks in long-running analysis
- GUI responsiveness during large batch operations

---

## [1.0.0] - 2025-11-29

### Added
- Core LIBS spectroscopic analysis engine
- Deep learning element detection (Informer model)
- Interactive visualization with drag-select regions
- Baseline correction using Asymmetric Least Squares
- Peak detection with configurable parameters
- Publication-ready plot export (300 DPI PNG)
- Element confidence scoring
- Optional Abel deconvolution
- Batch file processing
- Parameter preset management
- Comprehensive documentation

### Features
- **GUI**: PySide6-based cross-platform interface
- **Processing**: NumPy/SciPy pipeline for spectral analysis
- **Visualization**: PyQtGraph for real-time interactive plots
- **ML**: PyTorch-based element prediction
- **Export**: High-quality PNG figures and Excel results

### Documentation
- README with quick start guide
- Architecture documentation
- Parameter reference guide
- Troubleshooting guide
- Developer setup instructions
- Contributing guidelines

---

## Version Numbering

```
X.Y.Z
│ │ └─── Patch: Bug fixes
│ └───── Minor: New features (backwards compatible)
└─────── Major: Breaking changes
```

**Examples:**
- 1.0.0 → 1.0.1: Bug fix
- 1.0.1 → 1.1.0: New feature
- 1.1.0 → 2.0.0: Major refactor/breaking change

---

## Release Process

1. Update version in `__version__` variable
2. Update CHANGELOG.md
3. Create git tag: `git tag -a v1.0.0 -m "Release 1.0.0"`
4. Push tag: `git push origin v1.0.0`
5. Create GitHub Release with notes

---

## Migration Guides

### From 0.x to 1.0.0

**Parameter Structure Change**:
```python
# Old (0.x)
result = run_full_analysis(spectrum, lambda=1e5, p=0.001)

# New (1.0.0)
result = run_full_analysis({
    'asc_content': spectrum,
    'baseline_lambda': 1e5,
    'baseline_p': 0.001,
    'analysis_mode': 'predict'
})
```

**API Changes**:
- `prepare_asc_data()` now returns normalized spectrum directly
- `predict_elements()` replaced with integrated prediction in `run_full_analysis()`
- New `run_batch_analysis()` function for multiple files

---

## Deprecated Features

**Currently Deprecated** (will be removed in v2.0.0):
- `app/worker.py` (legacy) - use `app/ui/worker.py` instead
- Direct model loading - use `load_assets()` instead

---

## Known Limitations

- Model supports max 18 element detection
- Wavelength range: 250-800 nm (typical LIBS range)
- CSV export limited to 1 million rows
- Abel deconvolution assumes cylindrical geometry

---

## Future Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features and timelines.

---

**Last Updated**: November 29, 2025

