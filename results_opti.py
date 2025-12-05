#!/usr/bin/env python3
import matplotlib.ticker as mtick
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
    "figure.dpi": 300,
})

BASE_DIR = "results"
OUT_DIR = "comparisons"
os.makedirs(OUT_DIR, exist_ok=True)

# all your models
MODELS = [
    ("canelita_full_01", 256, 1024),
    ("canelita_full_02", 256, 2048),
    ("canelita_full_03", 256, 1024),
    ("canelita_full_04", 256, 256),
    ("canelita_full_05", 128, 256),
    ("canelita_full_06", 256, 512),
    ("canelita_full_07", 256, 1024),
    ("canelita_full_08", 256, 1024),
    ("canelita_full_09", 128, 1024),
    ("canelita_full_10", 128, 256),
    ("canelita_full_11", 64, 1024),
    ("canelita_full_12", 64, 128)
]

results = []

for name, hid, k in MODELS:
    folder = os.path.join(BASE_DIR, name, "boc_ista_results_p_l")

    meta_path = os.path.join(folder, "model_meta.json")
    metrics_path = os.path.join(folder, "final_test_metrics.txt")

    if not os.path.exists(meta_path) or not os.path.exists(metrics_path):
        print("[WARN] Missing results for", name)
        continue

    # read meta
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # read accuracy
    with open(metrics_path, "r") as f:
        lines = f.read().strip().splitlines()
    acc = float(lines[0].split(":")[1].strip())

    results.append({
        "name": name,
        "hidden": hid,
        "k": k,
        "acc": acc,
        "p": meta["best_p"],
        "lambda": meta["best_lambda"]
    })

names = [r["name"].replace("canelita_full_", "model_") for r in results]
accs = np.array([r["acc"] for r in results])
ps = np.array([r["p"] for r in results])
lambdas = np.array([r["lambda"] for r in results])


def annotate_bars(ax):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.3f}",
                    (p.get_x() + p.get_width()/2, height),
                    ha='center', va='bottom', fontsize=11)


# sort by accuracy if you want (optional)
order = np.argsort(accs)[::-1]
names_ord = [names[i] for i in order]
accs_ord = accs[order]
ps_ord = ps[order]
lambdas_ord = lambdas[order]


# 1) Accuracy per model
plt.figure(figsize=(6.5, 4.7))
ax = plt.gca()

best_idx = np.argmax(accs_ord)

colors = ["royalblue"] * len(accs_ord)
colors[best_idx] = "#000E78"

bars = ax.bar(names_ord, accs_ord,
              color=colors, alpha=0.85, width=0.6)

# annotation smaller
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f"{height:.3f}",
                (p.get_x() + p.get_width()/2, height+0.005),
                ha='center', va='bottom', fontsize=10)

# axis labels
ax.set_ylabel("Test Accuracy", fontsize=12)
ax.set_title("Test Accuracy per VQ-VAE Model", fontsize=14)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# tick labels
plt.xticks(rotation=45, ha="right", fontsize=10)
ax.tick_params(axis='y', labelsize=10)

ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout(pad=0.2)
plt.savefig(f"{OUT_DIR}/acc_per_model.png",
            dpi=300, bbox_inches='tight')
plt.close()


# 2) Best p per model
plt.figure(figsize=(6.5, 4.7))
ax = plt.gca()

bars = ax.bar(
    names_ord,
    ps_ord,
    color="#FFFBE4",
    edgecolor="#B09B56",
    linewidth=1.2,
    alpha=0.9,
    width=0.6
)


ax.set_ylabel("Best $p$", fontsize=12)
ax.set_title("Best $p$ per Model", fontsize=14)

plt.xticks(rotation=45, ha="right", fontsize=10)
ax.tick_params(axis='y', labelsize=10)

ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout(pad=0.2)
plt.savefig(
    f"{OUT_DIR}/p_per_model.png",
    dpi=300,
    bbox_inches='tight'
)
plt.close()


# 3) Frequency of best lambda
unique_l, counts_l = np.unique(lambdas, return_counts=True)

plt.figure(figsize=(6.5, 4.5))
ax = plt.gca()

palette = ["#a3be8c", "#a3be8c", "#a3be8c", "#a3be8c", "#a3be8c"]

for i, (l, c) in enumerate(zip(unique_l, counts_l)):
    ax.bar(l, c,
           width=0.00005,
           color=palette[i % len(palette)],
           edgecolor="#333333",
           linewidth=0.8,
           alpha=0.9)

    ax.text(l, c + 0.05,
            f"λ={l}",
            ha='center', fontsize=10)

ax.set_title("Frequency of Best $\\lambda$", fontsize=14)
ax.set_xlabel("$\\lambda$", fontsize=12)
ax.set_ylabel("Count", fontsize=12)

plt.xticks(fontsize=10)
ax.tick_params(axis='y', labelsize=10)

ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout(pad=0.2)
plt.savefig(f"{OUT_DIR}/freq_lambda.png",
            dpi=300, bbox_inches='tight')
plt.close()
