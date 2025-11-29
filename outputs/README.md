# Outputs Directory

**Generated outputs dan results dari analysis**

## Contents

```
outputs/
├── plots/              # Generated visualization plots
├── results/            # Analysis results dan metrics
├── models/             # Exported model checkpoints (optional)
└── reports/            # Detailed analysis reports
```

## Generated Files

This folder contains:
- **Publication plots** (PNG, 300 DPI) dari GUI export
- **Model checkpoints** dari training
- **Analysis reports** dari evaluation scripts
- **Results summaries** dengan metrics

## Usage

Generated automatically by:
- `app/main.py` - GUI export functionality
- `scripts/eval.py` - Evaluation metrics
- `scripts/train.py` - Training checkpoints

## Paper Reference

Walidain, B., Idris, N., Saddami, K., Yuzza, N., & Mitaphonna, R. (2025).
"Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine."
IOP Conference Series: Earth and Environmental Science, AIC 2025. (in press)

## Note

This folder is typically in `.gitignore` as it contains generated/temporary files.
Keep important plots manually or add to version control as needed.

