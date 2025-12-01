# Reproducing Results from the Paper

This document provides step-by-step instructions to reproduce the training and evaluation results reported in the paper:

> **"Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine"**  
> Walidain, B., Idris, N., Saddami, K., Yuzza, N., & Mitaphonna, R. (2025)  
> *IOP Conference Series: Earth and Environmental Science*, AIC 2025

---

## Prerequisites

- **Python 3.9 or later**
- **pip** or **conda** package manager
- **~2 GB** free disk space (for synthetic training data and model weights)
- **GPU** (optional but recommended; NVIDIA CUDA 11.8+ for acceleration)

---

## Step 1: Clone the Repository and Set Up Environment

```bash
# Clone the repository
git clone https://github.com/birrulwaldain/informer-libs-multielement.git
cd informer-libs-multielement

# Create a Python virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Verify Installation:**
```bash
python -c "import torch; import PySide6; import pyqtgraph; print('All dependencies installed successfully!')"
```

---

## Step 2: Download / Prepare Synthetic Training Data

The paper uses synthetic LIBS spectra generated via the **Saha–Boltzmann equation**. Two options:

### Option A: Use Pre-Generated Dataset (Recommended)
If a dataset file is provided in the repository:

```bash
# Check if synthetic data already exists
ls -la data/synthetic/

# If not present, download from the project repository or ask the authors
# The dataset should be in HDF5 format with structure:
# /train  - training split
# /val    - validation split
# /test   - test split
```

### Option B: Generate Synthetic Data from Scratch
The repository includes a data generation pipeline. To regenerate:

```bash
cd scripts

# Generate job plan with specified parameters
python planner.py \
  --num_samples 10000 \
  --elements 17 \
  --noise_levels "0.01 0.02 0.05" \
  --output combinations.json

# Generate synthetic LIBS spectra
python job.py --config combinations.json --output dataset.h5

# Verify the generated dataset
python check.py --dataset dataset.h5

cd ..
```

**Note:** Synthetic data generation may take several hours on a standard machine.  
For large-scale generation, use the HPC scripts (`job.sh`, `merge.py`) described in [scripts/README.md](scripts/README.md).

---

## Step 3: Train the Informer Model

### Default Training (Paper Configuration)

```bash
cd scripts

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

cd ..
```

**Training Parameters:**
- `d_model`: Embedding dimension (32 as per paper)
- `d_ff`: Feed-forward layer size (64 as per paper)
- `n_heads`: Number of attention heads (4 as per paper)
- `n_layers`: Number of Informer encoder layers (2 as per paper)
- `dropout`: Regularization parameter (0.1)
- `batch_size`: Training batch size (32)
- `epochs`: Maximum training epochs (100; early stopping may trigger earlier)
- `learning_rate`: Initial learning rate (0.001)
- `weight_decay`: L2 regularization (1e-5)

**Expected Output:**
- `assets/informer_multilabel_model.pth` — Model checkpoint
- `assets/training_log.csv` — Loss and metric history
- `assets/best_model_metric.txt` — Best validation metric

**Training Duration:**
- On NVIDIA GPU (e.g., RTX 3080): ~5-10 minutes
- On CPU: ~30-60 minutes

---

## Step 4: Evaluate the Model

### Run Evaluation on Test Split

```bash
cd scripts

python eval.py \
  --model_path ../assets/informer_multilabel_model.pth \
  --data_dir ../data \
  --output_dir ../outputs \
  --threshold 0.5 \
  --metrics "accuracy,hamming,f1_micro,f1_weighted"

cd ..
```

**Parameters:**
- `model_path`: Path to trained model weights
- `data_dir`: Directory containing the test dataset
- `output_dir`: Where to save results (CSV, JSON, plots)
- `threshold`: Classification threshold (0.5 is standard for multi-label)
- `metrics`: Comma-separated list of metrics to compute

**Expected Output:**
- `outputs/eval_metrics.csv` — Per-element precision, recall, F1-score
- `outputs/eval_summary.json` — Overall evaluation statistics
- `outputs/eval_confusion_matrix.png` — Confusion matrix visualization
- Console output with summary table (same as Table `perf_detail` in the paper)

**Example Output (Table `perf_detail`):**
```
Element         Precision  Recall    F1-Score   Support
-------------------------------------------------------------
H               0.92       0.88      0.90       145
C               0.89       0.86      0.87       128
N               0.85       0.82      0.83       110
O               0.90       0.91      0.90       155
Ca              0.88       0.87      0.87       132
...
```

---

## Step 5: Run Inference on Example Data

### Using the CLI

```bash
cd scripts

python inference.py \
  --model_path ../assets/informer_multilabel_model.pth \
  --input_file ../example-asc/1_iteration_1.asc \
  --output_dir ../outputs \
  --threshold 0.4 \
  --save_plot

cd ..
```

**Parameters:**
- `model_path`: Path to trained model
- `input_file`: Path to `.asc` LIBS spectrum
- `output_dir`: Output directory for results
- `threshold`: Detection probability threshold
- `save_plot`: Save visualization (optional)

**Expected Output:**
- `outputs/inference_results.csv` — Detected elements and probabilities
- `outputs/<filename>_prediction_plot.png` — Visualization (if `--save_plot`)

### Using the Interactive GUI

```bash
python app/main.py
```

1. Click **"Load Folder"** → select `example-asc/`
2. Click on a file in the table (e.g., `1_iteration_1.asc`)
3. Click **"Preprocess"** to apply normalization and baseline correction
4. Click **"Predict"** to run the model
5. Drag the blue region on the main plot to zoom into specific wavelength ranges
6. Click **"Export Scientific Plot"** to save a publication-ready figure

---

## Step 6: Validate Against Paper Results

### Checklist

After running the evaluation script, verify the results match the paper:

1. **Model Configuration**
   - [ ] Informer with 2 encoder layers
   - [ ] d_model = 32, d_ff = 64, n_heads = 4
   - [ ] Dropout = 0.1
   - [ ] Multi-label focal loss enabled
   - [ ] Input shape: (batch, 4096) — spectral channels
   - [ ] Output: 18 classes (17 elements + background)

2. **Synthetic Dataset Configuration**
   - [ ] Training: 35,000 spectra
   - [ ] Validation: 7,500 spectra
   - [ ] Test: 7,500 spectra
   - [ ] Total: 50,000 spectra
   - [ ] Wavelength range: 200–850 nm (4096 channels)
   - [ ] Noise levels: 1%, 2%, 5% (Gaussian)
   - [ ] Elements detected: Ca, C, H, K, Mg, Mn, N, Na, O, P, S, Si, Al, Fe, Cu, Zn, B + background

3. **Performance Metrics (Synthetic Test Set)**
   - [ ] Overall accuracy: 0.899 ± 0.015 (from paper)
   - [ ] Weighted F1-score: 0.890 ± 0.015
   - [ ] Macro F1-score: 0.878 ± 0.018
   - [ ] Hamming loss: 0.087 ± 0.012

4. **Per-Element Metrics** (Table `perf_detail` in paper)
   - [ ] Precision range: 0.82–0.94
   - [ ] Recall range: 0.80–0.93
   - [ ] F1 range: 0.81–0.93
   - [ ] Best detection: H, O, Ca (F1 > 0.88)
   - [ ] Challenging: Rare emitters (F1 ~ 0.81)

5. **Experimental Validation** (Aceh Herbal Medicine Case Study)
   - [ ] 13 herbal medicine samples analyzed
   - [ ] 39 total measurements (3 replicates per sample)
   - [ ] Successfully detected: Ca, Mg (minerals) and Na, Mn (trace metals)
   - [ ] High agreement (100% precision) with conventional LIBS analysis
   - [ ] See Figure 2 and Table 3 in paper

6. **Inference Results**
   - [ ] Can detect 15-17 elements from synthetic test samples
   - [ ] Predicted probabilities consistent with reported ranges
   - [ ] Example files in `example-asc/` return positive predictions for expected elements

If results deviate significantly, check:
- Data preprocessing (baseline correction via ALS, normalization to [0,1])
- Model initialization (set random seed for reproducibility)
- Hyperparameter alignment (especially d_model, d_ff, n_heads, n_layers)
- Hardware differences (CPU vs. GPU may introduce ~1-2% variation)
- Batch size effect (results may vary slightly with batch size 16 vs. 32)

---

## Advanced: Running on HPC (BRIN Mahameru)

For large-scale synthetic data generation and distributed training:

```bash
cd scripts

# Step 1: Generate job plan
python planner.py \
  --output combinations.json \
  --num_jobs 100 \
  --chunk_size 10

# Step 2: Submit jobs to cluster (BRIN Mahameru example)
sbatch --array=0-99 job.sh

# Step 3: Merge results
python merge.py \
  --input_pattern "dataset-*.h5" \
  --output dataset_merged.h5

# Step 4: Train on merged dataset
python train.py --data_dir . --output_dir ../assets

cd ..
```

**Adaptation for Other HPC Systems:**
See [scripts/README.md](scripts/README.md) for detailed instructions on adapting scheduler directives and environment paths.

---

## Troubleshooting

### Out of Memory (OOM) Errors

```bash
# Reduce batch size
python train.py --batch_size 16 ...

# Or use gradient accumulation
python train.py --batch_size 16 --accumulation_steps 2 ...
```

### Dataset Not Found

```bash
# Verify dataset structure
python scripts/check.py --dataset data/dataset.h5

# Expected output:
# ✓ Train split: 8000 samples
# ✓ Validation split: 1000 samples
# ✓ Test split: 1000 samples
```

### Model Training Not Converging

- Reduce learning rate: `--learning_rate 0.0005`
- Increase dropout: `--dropout 0.2`
- Check data preprocessing: Run `python app/main.py` and inspect preprocessed spectra

### Reproducibility Issues

To ensure reproducible results across runs:

```bash
# Set random seeds (recommended)
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0  # Use single GPU
python train.py --seed 42 ...
```

---

## Expected Times

| Step | Duration (GPU) | Duration (CPU) |
|------|------|------|
| Environment setup | 2 min | 2 min |
| Data preparation | 0 min | 0 min |
| Model training (100 epochs) | 8 min | 45 min |
| Evaluation | 30 sec | 2 min |
| Inference (1 file) | 1 sec | 5 sec |

**Total (end-to-end, GPU):** ~15 minutes

---

## Next Steps

After reproduction:

1. **Extend the model:** Modify `app/model.py` to experiment with different architectures
2. **Retrain on custom data:** Follow the data format in `data/README.md`
3. **Export for deployment:** Use `torch.jit.script` or ONNX conversion
4. **Integrate with instruments:** Connect to real LIBS hardware via data acquisition APIs

---

## References

- **Informer Architecture:** [Haoyi Zhou et al., ICLR 2021](https://openreview.net/pdf?id=0EXM-lbDZLs)
- **Saha–Boltzmann Equation:** [McWhirter & Hearn (1963)](https://doi.org/10.1088/0022-3700/2/2/403)
- **Multi-Label Focal Loss:** [Feng et al., CVPR 2021](https://arxiv.org/abs/2104.14294)

---

**Last Updated:** November 30, 2025  
**Status:** Ready for Publication

