# 🫀 ISTA Optimization for Arrhythmia Classification

> Sparse logistic regression optimized with ISTA to classify ventricular vs. supraventricular tachycardia with bundle branch block from ECG signals.

---

## Overview

This repository contains the **optimization component** of a research project on binary arrhythmia classification:

| Class       | Description                                           |
| ----------- | ----------------------------------------------------- |
| **VT**      | Ventricular Tachycardia                               |
| **SVT+BBB** | Supraventricular Tachycardia with Bundle Branch Block |

The classifier uses **sparse logistic regression** trained via the **Iterative Shrinkage–Thresholding Algorithm (ISTA)**, operating on Bag-of-Codes representations extracted from pretrained VQ-VAE models.

---

## ⚠️ Important Note

Pretrained VQ-VAE models and raw ECG data are **not included**, as they belong to ongoing unpublished research. This repository focuses solely on the **optimization pipeline**.

---

## Repository Structure

```
.
├── opti_approach.py       # Main script: Bag-of-Codes loading, ISTA optimization, evaluation
├── results_opti.py        # Functions for plotting and summarizing results
├── run_all_models.sh      # Runs ISTA for all VQ-VAE model configurations
├── split_class_tvt.py     # Train/validation/test split without subject leakage
└── comparisons/           # Output figures generated from results_opti.py
```

---

## Method

```
ECG Signals
    └─► VQ-VAE Encoding → Bag-of-Codes representation
            └─► Sparse Logistic Regression
                    └─► ISTA Optimization
                            └─► VT / SVT+BBB Classification
```

The key design choices are:

- **Sparsity** via L1 regularization, enforced through the ISTA proximal operator
- **Subject-aware splits** to prevent data leakage across train/validation/test sets
- **Multi-configuration evaluation** across different VQ-VAE model settings

---

## Usage

### Run ISTA for all model configurations

```bash
bash run_all_models.sh
```

### Run a single optimization

```python
python opti_approach.py
```

### Plot and summarize results

```python
python results_opti.py
```

---

## License

This project is part of ongoing research. Please contact the authors before reusing any part of this code.
