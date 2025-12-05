ISTA Optimization for Arrhythmia Classification

This repository contains the optimization component of a research project on classifying:
- Ventricular Tachycardia (VT)
- Supraventricular Tachycardia with Bundle Branch Block (SVT+BBB)

The classifier is trained using sparse logistic regression optimized with ISTA (Iterative Shrinkage–Thresholding Algorithm).

⚠️ Note:
Pretrained VQ-VAE models and raw ECG data are not included, as they belong to ongoing unpublished research.
This repository focuses solely on the optimization pipeline.

Files structure:
opti_approach.py — Main script: Bag-of-Codes loading, ISTA optimization, evaluation 
results_opti.py — Functions for plotting and summarizing results 
run_all_models.sh — Runs ISTA for all VQ-VAE model configurations 
split_class_tvt.py — Train/validation/test split without subject leakage
comparisons/ - includes the .png generated from results_opti.py
