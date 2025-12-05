README
-------------------------------

This repository contains the optimization component of a project on classifying ventricular tachycardia (VT) vs. supraventricular tachycardia with bundle branch block (SVT+BBB).

Only the ISTA optimization pipeline is included.
Pretrained VQ-VAE models and ECG data are not shared because they belong to ongoing unpublished research.

--------------------------------

Contents

opti_approach.py — Main script: Bag-of-Codes loading, ISTA optimization, evaluation
results_opti.py — Functions for plotting and summarizing results
run_all_models.sh — Runs ISTA for all VQ-VAE model configurations
split_class_tvt.py — Train/validation/test split without subject leakage