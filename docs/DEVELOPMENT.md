# Development Setup Guide

Instructions for setting up a local development environment.

---

## Prerequisites

- **Git**: [Download](https://git-scm.com/)
- **Python 3.9+**: [Download](https://www.python.org/)
- **GitHub Account**: For fork and pull requests

---

## Installation

### 1. Clone Repository

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/yourusername/spectroscopic-analyzer.git
cd spectroscopic-analyzer

# Add upstream remote for sync
git remote add upstream https://github.com/originalauthor/spectroscopic-analyzer.git
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Verify activation (should show .venv prefix)
which python
```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install development tools
pip install -r requirements-dev.txt
```

### 4. Verify Installation

```bash
# Check imports work
python -c "import torch, PySide6, pyqtgraph, numpy; print('✓ All imports OK')"

# Run app (should open GUI)
python app/main.py
```

---

## Required vs Optional Dependencies

### Core Requirements (`requirements.txt`)

```
torch>=2.0.0           # Deep learning framework
PySide6>=6.4.0         # Qt GUI framework
pyqtgraph>=0.13.0      # Scientific plotting
numpy>=1.20.0          # Numerical computing
scipy>=1.8.0           # Scientific algorithms
pandas>=1.3.0          # Data manipulation
openpyxl>=3.8.0        # Excel export
PyAbel>=0.8.4          # Deconvolution (optional)
fastapi>=0.95.0        # API framework (optional)
uvicorn>=0.20.0        # ASGI server (optional)
```

### Development Requirements (`requirements-dev.txt`)

```
pytest>=7.0.0          # Testing framework
pytest-cov>=4.0.0      # Coverage reporting
flake8>=4.0.0          # Code linting
black>=22.0.0          # Code formatting
mypy>=0.990            # Type checking
sphinx>=4.5.0          # Documentation
sphinx-rtd-theme>=1.0  # Documentation theme
```

---

## Project Structure for Development

```
spectroscopic-analyzer/
├── app/                    # Source code
│   ├── __init__.py
│   ├── main.py             # Application entry
│   ├── model.py            # Neural network
│   ├── processing.py       # Data preprocessing
│   ├── core/
│   │   ├── __init__.py
│   │   └── analysis.py     # Analysis pipeline
│   └── ui/
│       ├── __init__.py
│       ├── main_window.py
│       ├── control_panel.py
│       ├── results_panel.py
│       ├── batch_dialog.py
│       └── worker.py
├── assets/                 # Model and data files
│   ├── informer_multilabel_model.pth
│   ├── element-map-18a.json
│   └── wavelengths_grid.json
├── example-asc/            # Example data for testing
│   └── *.asc
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── test_processing.py
│   ├── test_analysis.py
│   └── test_model.py
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md
│   ├── PARAMETERS.md
│   └── API.md
├── .github/
│   ├── workflows/
│   │   └── tests.yml
│   └── pull_request_template.md
├── requirements.txt
├── requirements-dev.txt    # ← Run: pip install -r this
├── README.md
├── LICENSE
└── .gitignore
```

---

## Code Style

### Python Formatting with Black

```bash
# Format entire project
black app/

# Check formatting without changes
black --check app/

# Format with custom line length
black --line-length=100 app/
```

### Linting with Flake8

```bash
# Run linting
flake8 app/ --max-line-length=100

# Show statistics
flake8 app/ --statistics --count

# Ignore specific rules
flake8 app/ --ignore=E203,W503
```

### Type Checking with mypy

```bash
# Check types
mypy app/

# Ignore missing library stubs
mypy app/ --ignore-missing-imports
```

### Pre-commit Hook (Optional)

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
black app/ || exit 1
flake8 app/ --max-line-length=100 || exit 1
pytest tests/ -q || exit 1
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

---

## Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run single file
pytest tests/test_analysis.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test
pytest tests/test_analysis.py::test_baseline_correction -v
```

### Writing Tests

**File: `tests/test_processing.py`**

```python
import pytest
import numpy as np
from app.processing import prepare_asc_data

class TestDataProcessing:
    """Test data preprocessing functions."""
    
    def test_prepare_asc_data_basic(self):
        """Test basic ASC data preparation."""
        # Arrange
        asc_content = "250.0 100\n251.0 200\n252.0 150"
        target_wl = np.linspace(250, 252, 100)
        
        # Act
        result = prepare_asc_data(
            asc_content,
            target_wl,
            target_max_intensity=0.8
        )
        
        # Assert
        assert result.shape == (100,)
        assert np.max(result) <= 0.8
        assert np.min(result) >= 0.0
    
    def test_prepare_asc_data_normalization(self):
        """Test intensity normalization."""
        asc_content = "250.0 1000\n251.0 2000\n252.0 1500"
        target_wl = np.linspace(250, 252, 50)
        
        result = prepare_asc_data(asc_content, target_wl, target_max_intensity=1.0)
        
        assert np.max(result) == pytest.approx(1.0)
```

---

## Documentation

### Building Documentation Locally

```bash
# Install sphinx
pip install sphinx sphinx-rtd-theme

# Build docs (requires docs/conf.py)
cd docs/
make html

# View in browser
open _build/html/index.html
```

### Writing Docstrings

Use NumPy style:

```python
def run_full_analysis(input_data: dict) -> dict:
    """Run complete spectroscopic analysis pipeline.
    
    Comprehensive analysis of LIBS data including preprocessing,
    peak detection, and element identification.
    
    Parameters
    ----------
    input_data : dict
        Configuration dictionary with keys:
        - 'asc_content' (str): Raw ASC file content
        - 'baseline_lambda' (float): ALS smoothness
        - 'peak_height' (float): Minimum peak height
        
    Returns
    -------
    dict
        Result dictionary with keys:
        - 'status' (str): 'success' or 'error'
        - 'preprocessed' (ndarray): Processed spectrum
        - 'detected_elements' (dict): Element probabilities
        - 'error' (str): Error message if failed
        
    Raises
    ------
    ValueError
        If input_data is missing required keys
    RuntimeError
        If model inference fails
        
    Examples
    --------
    >>> result = run_full_analysis({
    ...     'asc_content': open('sample.asc').read(),
    ...     'analysis_mode': 'predict'
    ... })
    >>> elements = result['detected_elements']
    
    References
    ----------
    .. [1] Eilers, P. H., & Boelens, H. F. (2005).
           Baseline correction with asymmetric least squares smoothing.
    """
    pass
```

---

## Debugging

### Using pdb (Python Debugger)

```python
# Add to code
import pdb; pdb.set_trace()  # Breakpoint

# Commands in debugger
(Pdb) step      # Step into function
(Pdb) next      # Next line
(Pdb) continue  # Continue execution
(Pdb) list      # Show code
(Pdb) print x   # Print variable
```

### Using VS Code Debugger

`.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Main App",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/app/main.py",
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}
```

### Print Debugging

```python
# Log to console during analysis
print(f"DEBUG: spectrum shape = {spectrum.shape}")
print(f"DEBUG: baseline range = [{baseline.min():.3f}, {baseline.max():.3f}]")

# Use logging module (better practice)
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Processing spectrum with {len(spectrum)} points")
```

---

## Common Development Tasks

### Adding New Analysis Feature

1. **Implement Core Logic**
   ```python
   # app/core/analysis.py
   def new_feature(spectrum, param1=default1):
       """New feature implementation."""
       result = ...
       return result
   ```

2. **Add UI Controls**
   ```python
   # app/ui/control_panel.py
   self.param1_input = QtWidgets.QLineEdit()
   self.param1_label = QtWidgets.QLabel("Parameter 1:")
   ```

3. **Wire Signal/Slot**
   ```python
   # app/ui/main_window.py
   self.button.clicked.connect(self.worker.run_feature)
   self.worker.featureFinished.connect(self.on_feature_done)
   ```

4. **Write Tests**
   ```python
   # tests/test_new_feature.py
   def test_new_feature():
       result = new_feature(test_spectrum)
       assert result is not None
   ```

### Fixing a Bug

1. **Identify Issue**
   ```bash
   # Read error message and traceback
   # Check logs in UI
   ```

2. **Create Branch**
   ```bash
   git checkout -b bugfix/issue-description
   ```

3. **Write Test for Bug**
   ```python
   def test_bug_reproduction():
       """Test that demonstrates the bug."""
       assert buggy_function(input) == expected_output
   ```

4. **Fix Code**
   ```python
   def buggy_function(input):
       # Fix goes here
       return correct_output
   ```

5. **Verify Test Passes**
   ```bash
   pytest tests/test_fix.py -v
   ```

6. **Create Pull Request**
   ```bash
   git push origin bugfix/issue-description
   # Go to GitHub and create PR
   ```

---

## Performance Profiling

### Using cProfile

```python
import cProfile
import pstats
from app.core.analysis import run_full_analysis

# Profile function
profiler = cProfile.Profile()
profiler.enable()

result = run_full_analysis(input_data)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Using line_profiler

```bash
pip install line_profiler

# Add @profile decorator
# @profile
# def slow_function():
#     ...

# Run
kernprof -l -v app/core/analysis.py
```

---

## Troubleshooting Development Setup

### Qt Platform Plugin Error

```
Could not find the Qt platform plugin
```

**Solution**:
```bash
# Reinstall PySide6
pip uninstall PySide6
pip install PySide6
```

### Model Loading Error

```
FileNotFoundError: assets/informer_multilabel_model.pth
```

**Solution**: Download model from releases or make sure it's in `assets/` folder.

### GPU Not Available

```python
import torch
print(torch.cuda.is_available())  # False
```

**Solution**:
```bash
# Install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Resources

- **PyTorch**: https://pytorch.org/docs/
- **PySide6**: https://doc.qt.io/qtforpython/
- **PyQtGraph**: http://www.pyqtgraph.org/
- **SciPy**: https://docs.scipy.org/
- **pytest**: https://docs.pytest.org/

---

**Last Updated**: November 2025

