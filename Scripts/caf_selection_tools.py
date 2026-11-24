#!/usr/bin/env python3
import os
import sys
import ROOT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

max_events = 250000 if len(sys.argv) <= 1 else (-1 if sys.argv[1].lower() == "all" else int(sys.argv[1]))
os.makedirs(os.path.expanduser("~/todays_pdfs"), exist_ok=True)
cache_path = os.path.expanduser(f"~/todays_pdfs/caf_{max_events}.pkl")

# 1. Load DataFrame from fast pickle cache if available, otherwise read from CAF .root file
if os.path.exists(cache_path):
    print(f"Loading cached DataFrame from {cache_path}...", flush=True)
    df = pd.read_pickle(cache_path)
else:
    ROOT.gROOT.SetBatch(True)
    for p in ["./build/lib", "../build/lib", "./build", "../build"]:
        ROOT.gSystem.AddDynamicPath(p)
    ROOT.gSystem.Load("libduneanaobj_StandardRecord")

    caf_path = "/storage/1/st15719/dune/atmo/Mach3_Inputs/CAFs/caf_new_sum.2.6M_weighted.root"
    print(f"Opening CAF file: {caf_path}", flush=True)
    f = ROOT.TFile.Open(caf_path, "READ")
    caf_tree = f.Get("cafTree")

    var_labels = [
        "true_pdg", "true_energy", "true_n_prim_mu", "true_iscc",
        "true_mom_x", "true_mom_y", "true_mom_z",
        "true_vtx_x", "true_vtx_y", "true_vtx_z",
        "reco_ok", "reco_n_prim",
        "reco_energy_lep_calo", "reco_energy_e_calo", "reco_energy_calo",
        "reco_energy_mu_range", "reco_energy_mu_mcs",
        "dir_lngtrk_x", "dir_lngtrk_y", "dir_lngtrk_z",
        "dir_heshw_x", "dir_heshw_y", "dir_heshw_z",
        "cvn_nue", "cvn_numu", "cvn_nc"
    ]

    rows = []
    print(f"Reading {max_events if max_events > 0 else 'ALL'} events into DataFrame...", flush=True)
    for n_proc, entry in enumerate(caf_tree):
        if 0 < max_events <= n_proc:
            break
        if entry.rec.mc.nu.size() != 1:
            continue

        true_nu = entry.rec.mc.nu[0]
        true_n_mu = sum(1 for p in true_nu.prim if abs(p.pdg) == 13)

        if entry.rec.common.ixn.pandora.size() == 1:
            r = entry.rec.common.ixn.pandora[0]
            reco_n_prim = sum(1 for p in r.part.pandora if p.primary)
            reco_part = [
                True, reco_n_prim,
                float(r.Enu.lep_calo), float(r.Enu.e_calo), float(r.Enu.calo),
                float(r.Enu.mu_range), float(r.Enu.mu_mcs),
                float(r.dir.lngtrk.x), float(r.dir.lngtrk.y), float(r.dir.lngtrk.z),
                float(r.dir.heshw.x), float(r.dir.heshw.y), float(r.dir.heshw.z),
                float(r.nuhyp.cvn.nue), float(r.nuhyp.cvn.numu), float(r.nuhyp.cvn.nc)
            ]
        else:
            reco_part = [False] + [np.nan] * 15

        rows.append([
            int(true_nu.pdg), float(true_nu.E), true_n_mu, bool(true_nu.iscc),
            float(true_nu.momentum.x), float(true_nu.momentum.y), float(true_nu.momentum.z),
            float(true_nu.vtx.x), float(true_nu.vtx.y), float(true_nu.vtx.z),
            *reco_part
        ])
        if (n_proc + 1) % 50000 == 0:
            print(f"  Processed {n_proc + 1} events...", flush=True)

    df = pd.DataFrame(rows, columns=var_labels)
    df.to_pickle(cache_path)
    print(f"Saved DataFrame cache to {cache_path} (shape: {df.shape})", flush=True)

# 2. Derived columns: Normalized true direction, Fiducial flag, Valid Vector flags & Residuals
p_mag = np.sqrt(df["true_mom_x"]**2 + df["true_mom_y"]**2 + df["true_mom_z"]**2)
df["true_dir_x"] = np.where(p_mag > 0, df["true_mom_x"] / p_mag, np.nan)
df["true_dir_y"] = np.where(p_mag > 0, df["true_mom_y"] / p_mag, np.nan)
df["true_dir_z"] = np.where(p_mag > 0, df["true_mom_z"] / p_mag, np.nan)

df["in_fiducial"] = (
    (df["true_vtx_x"].abs() < 310.0) &
    (df["true_vtx_y"].abs() < 550.0) &
    (df["true_vtx_z"] > 50.0) & (df["true_vtx_z"] < 1350.0)
)

# Filter out (0, 0, 0) placeholder direction vectors
trk_mag2 = df["dir_lngtrk_x"]**2 + df["dir_lngtrk_y"]**2 + df["dir_lngtrk_z"]**2
shw_mag2 = df["dir_heshw_x"]**2  + df["dir_heshw_y"]**2  + df["dir_heshw_z"]**2
df["has_valid_track"]  = (trk_mag2 > 0.01) & (trk_mag2 < 1.5)
df["has_valid_shower"] = (shw_mag2 > 0.01) & (shw_mag2 < 1.5)

df["frac_res_lep_calo"] = np.where((df["reco_energy_lep_calo"] > 0) & (df["true_energy"] > 0),
                                   (df["reco_energy_lep_calo"] - df["true_energy"]) / df["true_energy"], np.nan)
df["frac_res_e_calo"]   = np.where((df["reco_energy_e_calo"] > 0) & (df["true_energy"] > 0),
                                   (df["reco_energy_e_calo"] - df["true_energy"]) / df["true_energy"], np.nan)

for comp in ["x", "y", "z"]:
    df[f"res_lngtrk_{comp}"] = np.where(df["has_valid_track"],  df[f"dir_lngtrk_{comp}"] - df[f"true_dir_{comp}"], np.nan)
    df[f"res_heshw_{comp}"]  = np.where(df["has_valid_shower"], df[f"dir_heshw_{comp}"]  - df[f"true_dir_{comp}"], np.nan)

dot_trk = (df["dir_lngtrk_x"] * df["true_dir_x"] + df["dir_lngtrk_y"] * df["true_dir_y"] + df["dir_lngtrk_z"] * df["true_dir_z"])
dot_shw = (df["dir_heshw_x"]  * df["true_dir_x"] + df["dir_heshw_y"]  * df["true_dir_y"] + df["dir_heshw_z"]  * df["true_dir_z"])

df["angle_err_lngtrk_deg"] = np.where(df["has_valid_track"],  np.degrees(np.arccos(np.clip(dot_trk, -1.0, 1.0))), np.nan)
df["angle_err_heshw_deg"]  = np.where(df["has_valid_shower"], np.degrees(np.arccos(np.clip(dot_shw, -1.0, 1.0))), np.nan)

# ==============================================================================
# HELPER: MULTI-CONDITION DATAFRAME INDEX SELECTION
# ==============================================================================
def select_indices(dataframe, conditions_list, label=""):
    combined_mask = pd.Series(True, index=dataframe.index)
    for cond in conditions_list:
        combined_mask = combined_mask & cond
    matched_idx = dataframe.index[combined_mask]
    print(f"[{label}] Matched {len(matched_idx)} events | Sample indices: {matched_idx[:6].tolist()}", flush=True)
    return matched_idx

# TOOL 1: True Muons vs True Electrons (Fiducial, E > 0.5 GeV)
idx_t1_numu = select_indices(df, [
    df["reco_ok"] == True, df["true_iscc"] == True, df["true_pdg"].abs() == 14,
    df["in_fiducial"] == True, df["reco_energy_lep_calo"] > 0.5
], label="Tool 1: True nu_mu CC")

idx_t1_nue = select_indices(df, [
    df["reco_ok"] == True, df["true_iscc"] == True, df["true_pdg"].abs() == 12,
    df["in_fiducial"] == True, df["reco_energy_e_calo"] > 0.5
], label="Tool 1: True nu_e CC")

# Report % of valid tracks vs valid showers
df_t1_mu, df_t1_el = df.loc[idx_t1_numu], df.loc[idx_t1_nue]
print(f"  -> True nu_mu CC valid track: {100*df_t1_mu['has_valid_track'].mean():.1f}% | valid shower: {100*df_t1_mu['has_valid_shower'].mean():.1f}%", flush=True)
print(f"  -> True nu_e  CC valid track: {100*df_t1_el['has_valid_track'].mean():.1f}% | valid shower: {100*df_t1_el['has_valid_shower'].mean():.1f}%", flush=True)

# TOOL 2: Selected High-E Events (E_reco > 1 GeV & CVN >= 0.5)
idx_t2_numu = select_indices(df, [
    df["reco_ok"] == True, df["true_iscc"] == True, df["true_pdg"].abs() == 14,
    df["reco_energy_lep_calo"] > 1.0, df["cvn_numu"] >= 0.5
], label="Tool 2: Selected nu_mu CC")

idx_t2_nue = select_indices(df, [
    df["reco_ok"] == True, df["true_iscc"] == True, df["true_pdg"].abs() == 12,
    df["reco_energy_e_calo"] > 1.0, df["cvn_nue"] >= 0.5
], label="Tool 2: Selected nu_e CC")

# TOOL 3: Reconstructed Zenith Angle (dir_lngtrk_z) + Energy Cuts on True nu_mu CC
idx_t3_up_hi = select_indices(df, [
    df["reco_ok"] == True, df["true_iscc"] == True, df["true_pdg"].abs() == 14,
    df["has_valid_track"] == True, df["dir_lngtrk_z"] < -0.2, df["reco_energy_lep_calo"] > 1.0
], label="Tool 3: Reco Up-going (cos_z_reco < -0.2) & E > 1 GeV")

idx_t3_dn_hi = select_indices(df, [
    df["reco_ok"] == True, df["true_iscc"] == True, df["true_pdg"].abs() == 14,
    df["has_valid_track"] == True, df["dir_lngtrk_z"] > 0.2, df["reco_energy_lep_calo"] > 1.0
], label="Tool 3: Reco Down-going (cos_z_reco > 0.2) & E > 1 GeV")

idx_t3_up_lo = select_indices(df, [
    df["reco_ok"] == True, df["true_iscc"] == True, df["true_pdg"].abs() == 14,
    df["has_valid_track"] == True, df["dir_lngtrk_z"] < -0.2,
    df["reco_energy_lep_calo"] <= 1.0, df["reco_energy_lep_calo"] > 0.1
], label="Tool 3: Reco Up-going (cos_z_reco < -0.2) & 0.1 < E <= 1 GeV")

# ==============================================================================
# GENERATE UPDATED 4-PAGE PDF
# ==============================================================================
pdf_path = os.path.expanduser("~/todays_pdfs/CAF_Selection_Tools_v2.pdf")

with PdfPages(pdf_path) as pdf:
    # PAGE 1 (Tool 1): 3D Direction Distributions (Valid Vectors Only)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    dir_bins = np.linspace(-1.0, 1.0, 51)
    for i, comp in enumerate(["x", "y", "z"]):
        axes[0, i].hist(df_t1_mu[f"true_dir_{comp}"].dropna(), bins=dir_bins, histtype="step", lw=2, color="black", label="True Direction")
        axes[0, i].hist(df_t1_mu.loc[df_t1_mu["has_valid_track"], f"dir_lngtrk_{comp}"], bins=dir_bins, histtype="step", lw=1.8, color="royalblue", label="dir.lngtrk (valid)")
        axes[0, i].hist(df_t1_mu.loc[df_t1_mu["has_valid_shower"], f"dir_heshw_{comp}"], bins=dir_bins, histtype="step", lw=1.8, ls="--", color="darkorange", label="dir.heshw (valid)")
        axes[0, i].set_title(f"Tool 1 (True $\\nu_\\mu$ CC): Direction {comp.upper()} (Valid Only)")
        axes[0, i].set_xlabel(f"Unit Direction {comp}"); axes[0, i].set_ylabel("Events"); axes[0, i].legend(fontsize=8); axes[0, i].grid(alpha=0.3)

        axes[1, i].hist(df_t1_el[f"true_dir_{comp}"].dropna(), bins=dir_bins, histtype="step", lw=2, color="black", label="True Direction")
        axes[1, i].hist(df_t1_el.loc[df_t1_el["has_valid_track"], f"dir_lngtrk_{comp}"], bins=dir_bins, histtype="step", lw=1.8, ls="--", color="royalblue", label="dir.lngtrk (valid)")
        axes[1, i].hist(df_t1_el.loc[df_t1_el["has_valid_shower"], f"dir_heshw_{comp}"], bins=dir_bins, histtype="step", lw=1.8, color="crimson", label="dir.heshw (valid)")
        axes[1, i].set_title(f"Tool 1 (True $\\nu_e$ CC): Direction {comp.upper()} (Valid Only)")
        axes[1, i].set_xlabel(f"Unit Direction {comp}"); axes[1, i].set_ylabel("Events"); axes[1, i].legend(fontsize=8); axes[1, i].grid(alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # PAGE 2 (Tool 1b): Component Direction Residuals (Valid Vectors Only)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    res_d_bins = np.linspace(-1.0, 1.0, 61)
    for i, comp in enumerate(["x", "y", "z"]):
        axes[0, i].hist(df_t1_mu[f"res_lngtrk_{comp}"].dropna(), bins=res_d_bins, histtype="step", lw=2, color="royalblue", label="lngtrk - true")
        axes[0, i].hist(df_t1_mu[f"res_heshw_{comp}"].dropna(), bins=res_d_bins, histtype="step", lw=2, ls="--", color="darkorange", label="heshw - true")
        axes[0, i].axvline(0, color="k", ls=":", lw=1); axes[0, i].set_title(f"True $\\nu_\\mu$ CC: $\\Delta$dir.{comp} Residual")
        axes[0, i].set_xlabel(f"Reco {comp} - True {comp}"); axes[0, i].legend(fontsize=8); axes[0, i].grid(alpha=0.3)

        axes[1, i].hist(df_t1_el[f"res_lngtrk_{comp}"].dropna(), bins=res_d_bins, histtype="step", lw=2, ls="--", color="royalblue", label="lngtrk - true")
        axes[1, i].hist(df_t1_el[f"res_heshw_{comp}"].dropna(), bins=res_d_bins, histtype="step", lw=2, color="crimson", label="heshw - true")
        axes[1, i].axvline(0, color="k", ls=":", lw=1); axes[1, i].set_title(f"True $\\nu_e$ CC: $\\Delta$dir.{comp} Residual")
        axes[1, i].set_xlabel(f"Reco {comp} - True {comp}"); axes[1, i].legend(fontsize=8); axes[1, i].grid(alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # PAGE 3 (Tool 2): 3D Opening Angle & Energy Quality (Valid Vectors Only)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    df_t2_mu, df_t2_el = df.loc[idx_t2_numu], df.loc[idx_t2_nue]
    ang_bins = np.linspace(0, 90, 61)
    axes[0].hist(df_t2_mu["angle_err_lngtrk_deg"].dropna(), bins=ang_bins, density=True, histtype="step", lw=2, color="royalblue", label=r"Selected $\nu_\mu$ CC: lngtrk (valid)")
    axes[0].hist(df_t2_mu["angle_err_heshw_deg"].dropna(), bins=ang_bins, density=True, histtype="step", lw=1.8, ls="--", color="cornflowerblue", label=r"Selected $\nu_\mu$ CC: heshw (valid)")
    axes[0].hist(df_t2_el["angle_err_heshw_deg"].dropna(), bins=ang_bins, density=True, histtype="step", lw=2, color="crimson", label=r"Selected $\nu_e$ CC: heshw (valid)")
    axes[0].hist(df_t2_el["angle_err_lngtrk_deg"].dropna(), bins=ang_bins, density=True, histtype="step", lw=1.8, ls="--", color="salmon", label=r"Selected $\nu_e$ CC: lngtrk (valid)")
    axes[0].set_title(r"Tool 2: 3D Opening Angle ($E_{\mathrm{reco}}>1$ GeV, CVN$\geq 0.5$, Valid Vectors)")
    axes[0].set_xlabel(r"3D Opening Angle $\Delta\theta_{\mathrm{reco,true}}$ [Degrees]"); axes[0].set_ylabel("Normalized Events"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    ebins_res = np.linspace(-1.0, 1.0, 61)
    axes[1].hist(df_t2_mu[df_t2_mu["angle_err_lngtrk_deg"] < 15.0]["frac_res_lep_calo"].dropna(), bins=ebins_res, density=True, histtype="step", lw=2, color="royalblue", label=r"$\nu_\mu$ CC: Good Track ($\Delta\theta < 15^\circ$)")
    axes[1].hist(df_t2_mu[df_t2_mu["angle_err_lngtrk_deg"] > 30.0]["frac_res_lep_calo"].dropna(), bins=ebins_res, density=True, histtype="step", lw=2, ls="--", color="navy", label=r"$\nu_\mu$ CC: Poor Track ($\Delta\theta > 30^\circ$)")
    axes[1].hist(df_t2_el[df_t2_el["angle_err_heshw_deg"]  < 15.0]["frac_res_e_calo"].dropna(), bins=ebins_res, density=True, histtype="step", lw=2, color="crimson", label=r"$\nu_e$ CC: Good Shower ($\Delta\theta < 15^\circ$)")
    axes[1].axvline(0, color="k", ls=":", lw=1); axes[1].set_title("Tool 2: Energy Residuals Split by Angle Quality")
    axes[1].set_xlabel(r"Fractional Energy Residual $(E_{\mathrm{reco}} - E_{\mathrm{true}}) / E_{\mathrm{true}}$"); axes[1].set_ylabel("Normalized Events"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # PAGE 4 (Tool 3): Reconstructed Zenith Angle + Energy Cuts
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    df_t3_up_hi, df_t3_dn_hi, df_t3_up_lo = df.loc[idx_t3_up_hi], df.loc[idx_t3_dn_hi], df.loc[idx_t3_up_lo]
    axes[0].hist(df_t3_up_hi["frac_res_lep_calo"].dropna(), bins=ebins_res, density=True, histtype="step", lw=2, color="royalblue", label=r"Reco Up-going ($\cos\theta_{z,\mathrm{reco}} < -0.2$) & $E > 1$ GeV")
    axes[0].hist(df_t3_dn_hi["frac_res_lep_calo"].dropna(), bins=ebins_res, density=True, histtype="step", lw=2, ls="--", color="darkgreen", label=r"Reco Down-going ($\cos\theta_{z,\mathrm{reco}} > 0.2$) & $E > 1$ GeV")
    axes[0].hist(df_t3_up_lo["frac_res_lep_calo"].dropna(), bins=ebins_res, density=True, histtype="step", lw=2, color="darkorange", label=r"Reco Up-going ($\cos\theta_{z,\mathrm{reco}} < -0.2$) & $0.1 < E \leq 1$ GeV")
    axes[0].axvline(0, color="k", ls=":", lw=1); axes[0].set_title("Tool 3: Energy Residuals by Reco Zenith & Energy Cuts")
    axes[0].set_xlabel(r"Fractional Energy Residual $(E_{\mathrm{reco}} - E_{\mathrm{true}}) / E_{\mathrm{true}}$"); axes[0].set_ylabel("Normalized Events"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    z_res_bins = np.linspace(-1.0, 1.0, 61)
    axes[1].hist(df_t3_up_hi["res_lngtrk_z"].dropna(), bins=z_res_bins, density=True, histtype="step", lw=2, color="royalblue", label=r"Reco Up-going ($\cos\theta_{z,\mathrm{reco}} < -0.2$) & $E > 1$ GeV")
    axes[1].hist(df_t3_dn_hi["res_lngtrk_z"].dropna(), bins=z_res_bins, density=True, histtype="step", lw=2, ls="--", color="darkgreen", label=r"Reco Down-going ($\cos\theta_{z,\mathrm{reco}} > 0.2$) & $E > 1$ GeV")
    axes[1].hist(df_t3_up_lo["res_lngtrk_z"].dropna(), bins=z_res_bins, density=True, histtype="step", lw=2, color="darkorange", label=r"Reco Up-going ($\cos\theta_{z,\mathrm{reco}} < -0.2$) & $0.1 < E \leq 1$ GeV")
    axes[1].axvline(0, color="k", ls=":", lw=1); axes[1].set_title("Tool 3: Zenith Angle Residuals by Reco Zenith & Energy Cuts")
    axes[1].set_xlabel(r"Zenith Angle Residual $\cos\theta_{z,\mathrm{reco}} - \cos\theta_{z,\mathrm{true}}$"); axes[1].set_ylabel("Normalized Events"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

print(f"SUCCESS! Saved upgraded 4-page PDF to: {pdf_path}", flush=True)