#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

cache_path = os.path.expanduser("~/todays_pdfs/caf_250000.pkl")
print(f"Loading cached DataFrame from {cache_path}...", flush=True)
df = pd.read_pickle(cache_path)

# Filter for reconstructed events with valid CVN scores in [0, 1]
valid_cvn = (
    (df["reco_ok"] == True) &
    (df["cvn_numu"] >= 0.0) & (df["cvn_numu"] <= 1.0) &
    (df["cvn_nue"]  >= 0.0) & (df["cvn_nue"]  <= 1.0) &
    (df["cvn_nc"]   >= 0.0) & (df["cvn_nc"]   <= 1.0)
)
df_v = df.loc[valid_cvn].copy()

# Assign True Class: 0 = nu_mu CC, 1 = nu_e CC, 2 = NC (exclude rare nu_tau CC from 3x3 matrix)
true_class = np.full(len(df_v), -1, dtype=int)
true_class[(df_v["true_pdg"].abs() == 14) & (df_v["true_iscc"] == True)] = 0
true_class[(df_v["true_pdg"].abs() == 12) & (df_v["true_iscc"] == True)] = 1
true_class[(df_v["true_iscc"] == False)] = 2

df_v["true_class"] = true_class
df_3 = df_v[df_v["true_class"] >= 0].copy()

# Predicted Class = whichever of [cvn_numu, cvn_nue, cvn_nc] has the highest score
scores = df_3[["cvn_numu", "cvn_nue", "cvn_nc"]].to_numpy()
df_3["pred_class"] = np.argmax(scores, axis=1)

# Build 3x3 Confusion Matrix
labels = [r"$\nu_\mu$ CC", r"$\nu_e$ CC", "NC"]
cm_counts = np.zeros((3, 3), dtype=int)
for t in range(3):
    for p in range(3):
        cm_counts[t, p] = int(((df_3["true_class"] == t) & (df_3["pred_class"] == p)).sum())

cm_norm = cm_counts / cm_counts.sum(axis=1, keepdims=True) * 100.0

print("\nConfusion Matrix (Raw Counts):")
print(pd.DataFrame(cm_counts, index=["True nu_mu CC", "True nu_e CC", "True NC"],
                   columns=["Pred nu_mu", "Pred nu_e", "Pred NC"]))
print("\nConfusion Matrix (Row-Normalized %):")
print(pd.DataFrame(np.round(cm_norm, 1), index=["True nu_mu CC", "True nu_e CC", "True NC"],
                   columns=["Pred nu_mu %", "Pred nu_e %", "Pred NC %"]))

# Helper for ROC Curve & AUC (pure NumPy, no sklearn required)
def compute_roc(y_true_binary, y_score, n_steps=200):
    thresholds = np.linspace(1.0, 0.0, n_steps)
    pos = (y_true_binary == 1)
    neg = (y_true_binary == 0)
    n_pos, n_neg = pos.sum(), neg.sum()
    tpr = np.array([(y_score[pos] >= th).sum() / n_pos for th in thresholds])
    fpr = np.array([(y_score[neg] >= th).sum() / n_neg for th in thresholds])
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, auc

fpr_numu, tpr_numu, auc_numu = compute_roc((df_3["true_class"] == 0).astype(int).to_numpy(), df_3["cvn_numu"].to_numpy())
fpr_nue,  tpr_nue,  auc_nue  = compute_roc((df_3["true_class"] == 1).astype(int).to_numpy(), df_3["cvn_nue"].to_numpy())
fpr_nc,   tpr_nc,   auc_nc   = compute_roc((df_3["true_class"] == 2).astype(int).to_numpy(), df_3["cvn_nc"].to_numpy())

print(f"\nROC AUC Scores -> cvn.numu: {auc_numu:.3f} | cvn.nue: {auc_nue:.3f} | cvn.nc: {auc_nc:.3f}", flush=True)

pdf_path = os.path.expanduser("~/todays_pdfs/CAF_CVN_Confusion_and_ROC.pdf")
with PdfPages(pdf_path) as pdf:
    # PAGE 1: Confusion Matrices (Counts & Row-Normalized %)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, mat, title, fmt, vmax in [
        (axes[0], cm_counts, "CVN Confusion Matrix (Event Counts)", "d", None),
        (axes[1], cm_norm,   "CVN Confusion Matrix (Row-Normalized %)", ".1f", 100.0)
    ]:
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Predicted Class (Highest CVN Score)", fontsize=11)
        ax.set_ylabel("True Interaction Type", fontsize=11)
        ax.set_title(title, fontsize=12)
        thresh = mat.max() * 0.55
        for i in range(3):
            for j in range(3):
                val_str = f"{mat[i, j]:{fmt}}" + ("%" if fmt == ".1f" else "")
                ax.text(j, i, val_str, ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if mat[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # PAGE 2: Multi-Class ROC Curves (cvn.numu, cvn.nue, cvn.nc)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.plot(fpr_numu, tpr_numu, lw=2.5, color="royalblue", label=f"cvn.numu (True $\\nu_\\mu$ CC) — AUC = {auc_numu:.3f}")
    ax.plot(fpr_nue,  tpr_nue,  lw=2.5, color="crimson",   label=f"cvn.nue (True $\\nu_e$ CC) — AUC = {auc_nue:.3f}")
    ax.plot(fpr_nc,   tpr_nc,   lw=2.5, color="seagreen",  label=f"cvn.nc (True NC) — AUC = {auc_nc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.5, label="Random Classifier (AUC = 0.500)")
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate (Background Acceptance)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Signal Efficiency)", fontsize=11)
    ax.set_title("CVN Classifier ROC Curves (All 3 Flavors)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.35)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"SUCCESS! Saved 2-page PDF to: {pdf_path}", flush=True)