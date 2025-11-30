# Scripts Documentation

This directory contains all training, evaluation, and utility scripts for the Informer-LIBS project.

---

## Quick Index

| Script | Purpose | Category | Status |
|--------|---------|----------|--------|
| `train.py` | Train the Informer model | Core | Used in Paper |
| `eval.py` | Evaluate model performance | Core | Used in Paper |
| `check.py` | Verify dataset integrity | Utility | Used in Paper |
| `planner.py` | Generate synthetic data jobs | HPC | Used in Paper |
| `job.py` | Execute data generation (per-node) | HPC | Used in Paper |
| `merge.py` | Merge HDF5 shards | HPC | Used in Paper |
| `map.py` | Element mapping utilities | Utility | Helper |
| `conv.py` | Data conversion tools | Utility | Helper |
| `job.sh` | SLURM submission script | HPC | Cluster-specific |
| `run.sh` | Local execution wrapper | Helper | Testing |
| `run-eval.sh` | Evaluation execution wrapper | Helper | Testing |
| `job-merge.sh` | Merge execution wrapper | Helper | Testing |

---

## Core Scripts (Training & Evaluation)

### train.py

**Purpose:** Train the Informer-based multi-element LIBS classifier

**Usage:**
```bash
python train.py \
  --data_dir ../data \
  --output_dir ../assets \
  --model_name informer_multilabel_model \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --d_model 32 \
  --d_ff 64 \
  --n_heads 4 \
  --n_layers 2 \
  --dropout 0.1 \
  --weight_decay 1e-5
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `../data` | Path to dataset directory (must contain train/val/test HDF5 files) |
| `--output_dir` | `../assets` | Where to save model checkpoints |
| `--model_name` | `informer_multilabel_model` | Base name for saved model |
| `--epochs` | 100 | Maximum training epochs |
| `--batch_size` | 32 | Training batch size |
| `--learning_rate` | 0.001 | Initial learning rate (AdamW) |
| `--d_model` | 32 | Embedding dimension (paper value) |
| `--d_ff` | 64 | Feed-forward dimension (paper value) |
| `--n_heads` | 4 | Number of attention heads (paper value) |
| `--n_layers` | 2 | Number of Informer encoder layers (paper value) |
| `--dropout` | 0.1 | Dropout rate for regularization |
| `--weight_decay` | 1e-5 | L2 regularization strength |
| `--seed` | 42 | Random seed for reproducibility |
| `--device` | auto | 'cuda' or 'cpu' |

**Output:**
- `informer_multilabel_model.pth` — Best model checkpoint
- `informer_multilabel_model_latest.pth` — Latest model (for resuming)
- `training_log.csv` — Epoch-wise loss and metrics
- `best_model_metric.txt` — Best validation metric achieved

**Typical Runtime:**
- GPU (RTX 3080): ~8 minutes
- CPU: ~45 minutes

**Paper Configuration:**
All defaults match the paper exactly. Run without arguments to reproduce:
```bash
python train.py
```

---

### eval.py

**Purpose:** Evaluate a trained model on test set and export metrics

**Usage:**
```bash
python eval.py \
  --model_path ../assets/informer_multilabel_model.pth \
  --data_dir ../data \
  --output_dir ../outputs \
  --threshold 0.5 \
  --metrics "accuracy,hamming,f1_micro,f1_weighted"
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `../assets/informer_multilabel_model.pth` | Path to trained model weights |
| `--data_dir` | `../data` | Path to dataset directory |
| `--output_dir` | `../outputs` | Where to save evaluation results |
| `--threshold` | 0.5 | Classification threshold (0–1) |
| `--batch_size` | 32 | Evaluation batch size |
| `--metrics` | `accuracy,hamming,f1_micro,f1_weighted,f1_macro` | Metrics to compute |
| `--save_plots` | True | Generate confusion matrix & ROC curves |
| `--save_csv` | True | Export per-element metrics as CSV |

**Output Files:**
- `eval_metrics.csv` — Per-element precision, recall, F1-score
- `eval_summary.json` — Overall metrics (accuracy, Hamming, F1)
- `eval_confusion_matrix.png` — Visualization
- `eval_roc_curves.png` — ROC curves (if enabled)
- Console: Human-readable table matching Table `perf_detail` in paper

**Example Output:**
```
==============================================
EVALUATION RESULTS (Test Set)
==============================================

Overall Metrics:
  Accuracy:           0.899
  Hamming Loss:       0.087
  Macro F1:           0.878
  Micro F1:           0.891
  Weighted F1:        0.890

Per-Element Metrics:
Element         Precision  Recall    F1-Score   Support
-------------------------------------------------------------
Background      0.95       0.96      0.95       850
H               0.92       0.88      0.90       145
C               0.89       0.86      0.87       128
...
```

**Typical Runtime:** ~30 seconds (GPU), ~2 minutes (CPU)

---

### check.py

**Purpose:** Verify dataset integrity and consistency

**Usage:**
```bash
python check.py --dataset ../data/synthetic/dataset_train.h5
```

**Checks:**
- ✓ HDF5 file readable
- ✓ Required groups exist (train, val, test)
- ✓ Tensor shapes correct (N, 4096) for spectra
- ✓ Label shapes (N, 18) for multi-hot encoding
- ✓ Wavelength grid monotonic 200–850 nm
- ✓ Intensity values in [0, 1]
- ✓ No NaN/Inf values
- ✓ Element distribution balanced
- ✓ Train/val/test splits non-overlapping

**Output:**
```
✓ dataset_train.h5 is valid
✓ Train set: 8000 spectra, 18 elements, balanced
✓ Validation set: 1000 spectra, OK
✓ Test set: 1000 spectra, OK
✓ Wavelength grid: [200.000, 850.000] nm, 4096 channels
✓ All checks passed
```

**Typical Runtime:** <5 seconds

---

## HPC-Oriented Scripts (Data Generation)

### planner.py

**Purpose:** Generate a plan file listing all synthetic data generation jobs

**Usage:**
```bash
python planner.py \
  --num_samples 10000 \
  --elements "17" \
  --noise_levels "0.01 0.02 0.05" \
  --temperature_range "5000 15000" \
  --output combinations.json
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_samples` | 10000 | Total synthetic spectra to generate |
| `--elements` | 17 | Number of elements to include |
| `--noise_levels` | "0.01 0.02 0.05" | Noise standard deviations to test |
| `--temperature_range` | "5000 15000" | Plasma temperature min max (K) |
| `--pressure_range` | "0.5 2.0" | Plasma pressure min max (atm) |
| `--output` | combinations.json | Output job plan file |

**Output:** `combinations.json` (text, ~10 MB for 10k samples)

**Format:**
```json
{
  "metadata": {
    "total_samples": 10000,
    "chunk_size": 100,
    "num_chunks": 100
  },
  "jobs": [
    {
      "chunk_id": 0,
      "num_samples": 100,
      "temperature": 5234,
      "pressure": 1.2,
      "noise": 0.02
    },
    ...
  ]
}
```

---

### job.py

**Purpose:** Execute synthetic LIBS data generation for a specific job chunk

**Usage:**
```bash
# Single job
python job.py --config combinations.json --chunk_id 0 --output dataset-chunk0.h5

# Or batch from shell script
for i in {0..99}; do
  python job.py --config combinations.json --chunk_id $i \
    --output dataset-chunk$i.h5
done
```

**Key Arguments:**

| Argument | Default | Description |
| `--config` | combinations.json | Path to job plan file (from planner.py) |
| `--chunk_id` | 0 | Which job to execute |
| `--output` | dataset.h5 | Output HDF5 file for this chunk |
| `--temp_override` | None | Override temperature (for testing) |
| `--seed` | auto | Random seed |

**Output:** `dataset-chunk<id>.h5` (HDF5)

**Structure:**
```
dataset-chunk0.h5
├── spectra      (100, 4096) float32
└── labels       (100, 18) int32
```

**Typical Runtime per Chunk:** ~2 minutes (100 samples)

---

### merge.py

**Purpose:** Merge multiple HDF5 shards into a single train/val/test dataset

**Usage:**
```bash
python merge.py \
  --input_pattern "dataset-chunk*.h5" \
  --output dataset_merged.h5 \
  --train_split 0.8 \
  --val_split 0.1
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_pattern` | dataset-chunk*.h5 | Glob pattern for HDF5 files to merge |
| `--output` | dataset_merged.h5 | Output merged HDF5 file |
| `--train_split` | 0.8 | Fraction for training (0.8) |
| `--val_split` | 0.1 | Fraction for validation (0.1) |
| `--test_split` | auto | Remainder for test (1 - train - val) |
| `--seed` | 42 | Random seed for reproducible split |

**Output:** `dataset_merged.h5`

**Structure:**
```
dataset_merged.h5
├── train/
│   ├── spectra      (8000, 4096)
│   └── labels       (8000, 18)
├── val/
│   ├── spectra      (1000, 4096)
│   └── labels       (1000, 18)
└── test/
    ├── spectra      (1000, 4096)
    └── labels       (1000, 18)
```

**Typical Runtime:** ~30 seconds

---

## Utility Scripts

### map.py

**Purpose:** Generate or update element mapping files

**Usage:**
```bash
python map.py --elements "H C N O Ca K Mg Mn Na Si" --output element-map.json
```

**Output:** `element-map.json`

```json
{
  "0": "background",
  "1": "H",
  "2": "C",
  ...
}
```

---

### conv.py

**Purpose:** Convert between data formats (ASC, CSV, HDF5)

**Usage:**
```bash
# ASC to HDF5
python conv.py --input_file spectrum.asc --input_format asc --output_format h5

# CSV to NumPy
python conv.py --input_file spectra.csv --input_format csv --output_format npy
```

---

## Helper Scripts (Shell Wrappers)

### job.sh

**Purpose:** SLURM batch submission script for HPC clusters

**Context:** Tailored to **BRIN's Mahameru cluster**. Requires adaptation for other systems.

**Typical SLURM Directives:**
```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=libs_job
```

**Usage (on Mahameru):**
```bash
# Submit array job
sbatch --array=0-99 job.sh

# Monitor
squeue -u $USER
```

**Adaptation for Other Clusters:**

| Item | Mahameru | Your Cluster |
|------|----------|---|
| Scheduler | SLURM | ? |
| GPU partition | `gpu` | ? |
| Module system | `module load` | ? |
| Conda activation | `conda activate libs` | ? |
| Data paths | `/home/$USER/data` | ? |

Edit `job.sh` to match your cluster's configuration.

---

### run.sh

**Purpose:** Local execution wrapper for training (testing/debugging)

```bash
./run.sh --epochs 10 --batch_size 16
```

**Features:**
- Activates virtual environment
- Sets PYTHONPATH
- Runs train.py with arguments
- Captures logs to `training.log`

---

### run-eval.sh

**Purpose:** Local execution wrapper for evaluation

```bash
./run-eval.sh --threshold 0.5
```

---

### job-merge.sh

**Purpose:** Wrapper for merging HDF5 shards

```bash
./job-merge.sh --input_pattern "dataset-chunk*.h5"
```

---

## Typical Workflows

### Workflow 1: Quick Test (5 min)
```bash
# Create small synthetic dataset
python planner.py --num_samples 100 --output tiny_plan.json
python job.py --config tiny_plan.json --output tiny.h5

# Train on it
python train.py --data_dir . --epochs 5 --batch_size 16

# Evaluate
python eval.py --model_path ../assets/informer_multilabel_model.pth
```

### Workflow 2: Full Reproduction (30 min on GPU)
```bash
# Generate full synthetic dataset
python planner.py --num_samples 10000 --output plan.json
for i in {0..99}; do
  python job.py --config plan.json --chunk_id $i --output chunk$i.h5 &
done
wait

# Merge
python merge.py --input_pattern "chunk*.h5" --output dataset.h5

# Train (default hyperparams match paper)
python train.py

# Evaluate
python eval.py
```

### Workflow 3: Distributed HPC (Mahameru)
```bash
# Plan
python planner.py --num_samples 100000 --output massive_plan.json

# Dispatch to cluster
sbatch --array=0-999 job.sh

# Wait for jobs to complete, then merge
python merge.py --input_pattern "dataset-chunk*.h5"

# Train on full dataset
python train.py --batch_size 64 --epochs 200
```

---

## Performance Notes

### Script Runtimes

| Script | Input Size | GPU | CPU |
|--------|-----------|-----|-----|
| planner.py | – | <1 sec | <1 sec |
| job.py (100 samples) | – | ~2 min | ~8 min |
| merge.py (100 shards) | 10k spectra | ~30 sec | ~1 min |
| train.py (10k spectra) | – | ~8 min | ~45 min |
| eval.py (1k spectra) | – | ~30 sec | ~2 min |

### Memory Requirements

| Operation | Peak Memory |
|-----------|---|
| Data generation (100 samples) | ~500 MB |
| Training (batch=32) | ~2 GB |
| Evaluation | ~1 GB |

---

## Troubleshooting

### Script Errors

**ImportError: No module named 'torch'**
```bash
pip install -r requirements.txt
```

**FileNotFoundError: Dataset not found**
```bash
# Generate synthetic data first
python planner.py && python job.py
```

**CUDA Out of Memory**
```bash
# Reduce batch size
python train.py --batch_size 16 --accumulation_steps 2
```

### HPC-Specific Issues

**Module not found on cluster**
```bash
module avail
module load cuda/11.8 python/3.10
```

**SLURM job fails**
```bash
# Check logs
scat log-<jobid>.out

# Debug interactively
srun --pty bash
```

---

## References

- **Informer Paper:** [Zhou et al., ICLR 2021](https://openreview.net/pdf?id=0EXM-lbDZLs)
- **LIBS Physics:** [Miziolek et al., Chemical Reviews 2006](https://doi.org/10.1021/cr010379f)
- **HDF5 Format:** [The HDF Group](https://www.h5py.org/)

---

## Contact & Support

**Questions about scripts?**
- Open an issue: [github.com/birrulwaldain/informer-libs-multielement/issues](https://github.com/birrulwaldain/informer-libs-multielement/issues)
- Email: birrul@mhs.usk.ac.id

**HPC-specific help?**
- Contact: nasrullah.idris@usk.ac.id
- Document cluster configuration and we'll help adapt scripts

---

**Last Updated:** November 30, 2025  
**Status:** Ready for Publication  
**Version:** 1.0.0

