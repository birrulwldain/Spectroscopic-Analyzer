# Repository Navigation Guide

Quick reference for finding documentation in the Informer-LIBS-Multielement repository.

---

## I want to...

### Get Started
- **Running the GUI**: See [README.md](README.md#quick-start) → Quick Start section
- **Install dependencies**: See [README.md](README.md#installation) → Installation
- **Understand what this is**: See [README.md](README.md#overview) → Overview section

### Reproduce the Paper
- **Follow exact steps**: Read [REPRODUCE.md](REPRODUCE.md)
- **Train the model**: See [REPRODUCE.md](REPRODUCE.md#step-3-train-the-informer-model)
- **Evaluate results**: See [REPRODUCE.md](REPRODUCE.md#step-4-evaluate-the-model)
- **Expected times**: See [REPRODUCE.md](REPRODUCE.md#expected-times)

### Understand the Model
- **Architecture details**: See [MODEL_CARD.md](MODEL_CARD.md#model-architecture)
- **Performance metrics**: See [MODEL_CARD.md](MODEL_CARD.md#model-performance)
- **Training configuration**: See [MODEL_CARD.md](MODEL_CARD.md#training-details)
- **Limitations**: See [MODEL_CARD.md](MODEL_CARD.md#limitations--biases)

### Work with Data
- **Get data**: See [data/README.md](data/README.md#data-availability--access)
- **Understand formats**: See [data/README.md](data/README.md#data-format)
- **Generate synthetic data**: See [data/README.md](data/README.md#generating-custom-training-data)
- **Preprocessing steps**: See [data/README.md](data/README.md#data-preprocessing-pipeline)

### Use Scripts
- **Overview of all scripts**: See [scripts/README.md](scripts/README.md)
- **Train the model**: See [scripts/README.md](scripts/README.md#trainpy)
- **Evaluate model**: See [scripts/README.md](scripts/README.md#evalpy)
- **Generate synthetic data**: See [scripts/README.md](scripts/README.md#jobpy)
- **HPC deployment**: See [scripts/README.md](scripts/README.md#hpc-oriented-scripts-data-generation)

### Use the GUI
- **Basic workflow**: See [README.md](README.md#basic-workflow)
- **Parameter controls**: See [app/ui/main_window.py](app/ui/main_window.py) (source code)
- **Troubleshooting GUI issues**: See [README.md](README.md#troubleshooting)

### Cite this Work
- **BibTeX**: See [README.md](README.md#bibtex-entry)
- **Human-readable**: See [README.md](README.md#human-readable-citation)
- **All formats**: See [CITATION.cff](CITATION.cff)

### Deploy to HPC
- **Cluster adaptation**: See [scripts/README.md](scripts/README.md#hpc-oriented-scripts-data-generation)
- **Job planning**: See [scripts/README.md](scripts/README.md#plannerpy)
- **Data generation on cluster**: See [scripts/README.md](scripts/README.md#jobpy)
- **SLURM script**: See [scripts/job.sh](scripts/job.sh)

### Fix Issues
- **Common problems**: See [README.md](README.md#troubleshooting)
- **GPU/CUDA issues**: See [README.md](README.md#cudagpu-not-detected)
- **Import errors**: See [README.md](README.md#import-errors)
- **Training issues**: See [REPRODUCE.md](REPRODUCE.md#troubleshooting)
- **Script errors**: See [scripts/README.md](scripts/README.md#troubleshooting)

### Extend the Research
- **Future work**: See [ROADMAP.md](ROADMAP.md)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Architecture details**: See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Implementation notes**: See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

### Find Contact Info
- **For paper-related questions**: See nasrullah.idris@usk.ac.id in [README.md](README.md#contact--support)
- **For code/GitHub issues**: See birrul@mhs.usk.ac.id in [README.md](README.md#contact--support)
- **GitHub issue tracker**: [github.com/.../issues](https://github.com/birrulwaldain/informer-libs-multielement/issues)

---

## Documentation Map

### Main Entry Points
- **README.md** — Overview and quick start
- **REPRODUCE.md** — Complete reproducibility guide
- **MODEL_CARD.md** — Technical model documentation
- **PUBLICATION_CHECKLIST.md** — What was done for publication

### Data & Scripts
- **data/README.md** — Data format and access
- **scripts/README.md** — All script documentation
- **example-asc/** — Example LIBS spectra for testing

### Technical Details
- **docs/ARCHITECTURE.md** — Application architecture
- **docs/DEVELOPMENT.md** — Development guidelines
- **docs/PARAMETERS.md** — Model parameters reference

### Administrative
- **CITATION.cff** — Citation metadata (CFF format)
- **LICENSE** — MIT License
- **CONTRIBUTING.md** — Contribution guidelines
- **ROADMAP.md** — Future plans
- **TROUBLESHOOTING.md** — Common issues

### Configuration
- **setup.py** — Python package setup
- **pyproject.toml** — Modern Python packaging
- **requirements.txt** — Dependencies
- **requirements-dev.txt** — Development dependencies

---

## Quick Links

### Key Sections

| Need | Link |
|------|------|
| Quick Start | [README.md#quick-start](README.md#quick-start) |
| Reproduce Paper | [REPRODUCE.md](REPRODUCE.md) |
| Model Details | [MODEL_CARD.md](MODEL_CARD.md) |
| Data Info | [data/README.md](data/README.md) |
| Script Help | [scripts/README.md](scripts/README.md) |
| How to Cite | [README.md#citation](README.md#citation) |
| Troubleshoot | [README.md#troubleshooting](README.md#troubleshooting) |
| Contact | [README.md#contact--support](README.md#contact--support) |

### Python Modules

| Module | File | Purpose |
|--------|------|---------|
| Model | [app/model.py](app/model.py) | Informer architecture |
| Processing | [app/processing.py](app/processing.py) | Data preprocessing |
| Analysis | [app/core/analysis.py](app/core/analysis.py) | Analysis pipeline |
| GUI | [app/ui/main_window.py](app/ui/main_window.py) | Interactive interface |
| Training | [scripts/train.py](scripts/train.py) | Model training |
| Evaluation | [scripts/eval.py](scripts/eval.py) | Model evaluation |

---

## File Organization

```
Root
├── Documentation
│   ├── README.md (START HERE)
│   ├── REPRODUCE.md
│   ├── MODEL_CARD.md
│   ├── PUBLICATION_CHECKLIST.md
│   ├── CITATION.cff
│   ├── LICENSE
│   ├── CONTRIBUTING.md
│   ├── ROADMAP.md
│   └── TROUBLESHOOTING.md
│
├── Application (app/)
│   ├── main.py (GUI entry point)
│   ├── model.py (Informer model)
│   ├── processing.py (Data preprocessing)
│   ├── core/ (Core modules)
│   └── ui/ (GUI components)
│
├── Scripts (scripts/)
│   ├── train.py (Training)
│   ├── eval.py (Evaluation)
│   ├── check.py (Data validation)
│   ├── planner.py (Job planning)
│   ├── job.py (Data generation)
│   ├── merge.py (Merge datasets)
│   └── README.md (Script docs)
│
├── Assets (assets/)
│   ├── informer_multilabel_model.pth (Weights)
│   ├── element-map-17.json (Element mapping)
│   └── wavelengths_grid.json (Wavelength grid)
│
├── Data (data/)
│   ├── synthetic/ (Training data - HDF5)
│   ├── experimental/ (Case study - on request)
│   └── README.md (Data docs)
│
├── Examples (example-asc/)
│   └── *.asc (Example LIBS spectra)
│
├── Technical (docs/)
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT.md
│   ├── PARAMETERS.md
│   └── openapi.yaml
│
└── Configuration
    ├── setup.py
    ├── pyproject.toml
    ├── requirements.txt
    └── requirements-dev.txt
```

---

## Common Tasks

### Task: Run the GUI
```bash
python app/main.py
```
Learn more: [README.md#quick-start](README.md#quick-start)

### Task: Train the Model
```bash
cd scripts
python train.py
```
Learn more: [REPRODUCE.md#step-3](REPRODUCE.md#step-3-train-the-informer-model)

### Task: Evaluate Model
```bash
cd scripts
python eval.py
```
Learn more: [REPRODUCE.md#step-4](REPRODUCE.md#step-4-evaluate-the-model)

### Task: Generate Synthetic Data
```bash
cd scripts
python planner.py --num_samples 10000
python job.py --config combinations.json
```
Learn more: [scripts/README.md#synthetic-data-generation](scripts/README.md)

### Task: Test Installation
```bash
python -c "import torch, PySide6, pyqtgraph; print('OK')"
```
Learn more: [REPRODUCE.md#prerequisites](REPRODUCE.md#prerequisites)

### Task: Reproduce Entire Paper
See [REPRODUCE.md](REPRODUCE.md) — complete step-by-step guide

---

## Search Tips

### If you see an error about...
- **Model weights**: See [assets/README.md](data/README.md#model-access)
- **Missing data**: See [data/README.md](data/README.md#data-availability--access)
- **Hyperparameters**: See [MODEL_CARD.md](MODEL_CARD.md#training-details)
- **Script usage**: See [scripts/README.md](scripts/README.md)
- **Import error**: See [README.md#import-errors](README.md#import-errors)

### If you want to know about...
- **Paper methodology**: See [README.md#overview](README.md#overview) and [MODEL_CARD.md](MODEL_CARD.md)
- **Data sources**: See [data/README.md](data/README.md)
- **Model architecture**: See [MODEL_CARD.md#model-architecture](MODEL_CARD.md#model-architecture)
- **Training procedure**: See [REPRODUCE.md#step-3](REPRODUCE.md#step-3-train-the-informer-model)
- **Evaluation metrics**: See [MODEL_CARD.md#model-performance](MODEL_CARD.md#model-performance)

---

## Last Updated

November 30, 2025

---

**Need help?** Open an [issue on GitHub](https://github.com/birrulwaldain/informer-libs-multielement/issues) or email birrul@mhs.usk.ac.id

