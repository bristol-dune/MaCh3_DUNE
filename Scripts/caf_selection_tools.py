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

# 1. Load DUNE StandardRecord library
ROOT.gROOT.SetBatch(True)
for p in ["./build/lib", "../build/lib", "./build", "../build"]:
    ROOT.gSystem.AddDynamicPath(p)
ROOT.gSystem.Load("libduneanaobj_StandardRecord")

caf_path = "/storage/1/st15719/dune/atmo/Mach3_Inputs/CAFs/caf_new_sum.2.6M_weighted.root"
print(f"Opening CAF file: {caf_path}", flush=True)
f = ROOT.TFile.Open(caf_path, "READ")
caf_tree = f.Get("cafTree")

max_events = 250000 if len(sys.argv) <= 1 else (-1 if sys.argv[1].lower() == "all" else int(sys.argv[1]))

# 2. Extract all truth and reconstructed variables into a Pandas DataFrame
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
print(f"DataFrame created with shape: {df.shape}", flush=True)

# 3. Derived columns: Normalized true direction, Fiducial flag, Residuals & Opening Angles
p_mag = np.sqrt(df["true_mom_x"]**2 + df["true_mom_y"]**2 + df["true_mom_z"]**2)
df["true_dir_x"] = np.where(p_mag > 0, df["true_mom_x"] / p_mag, np.nan)
df["true_dir_y"] = np.where(p_mag > 0, df["true_mom_y"] / p_mag, np.nan)
df["true_dir_z"] = np.where(p_mag > 0, df["true_mom_z"] / p_mag, np.nan)  # Zenith cos(theta_z)

df["in_fiducial"] = (
    (df["true_vtx_x"].abs() < 310.0) &
    (df["true_vtx_y"].abs() < 550.0) &
    (df["true_vtx_z"] > 50.0) & (df["true_vtx_z"] < 1350.0)
)

# Fractional energy residuals: (reco - true) / true
df["frac_res_lep_calo"] = np.where((df["reco_energy_lep_calo"] > 0) & (df["true_energy"] > 0),
                                   (df["reco_energy_lep_calo"] - df["true_energy"]) / df["true_energy"], np.nan)
df["frac_res_e_calo"]   = np.where((df["reco_energy_e_calo"] > 0) & (df["true_energy"] > 0),
                                   (df["reco_energy_e_calo"] - df["true_energy"]) / df["true_energy"], np.nan)

# Direction residuals (reco - true) for Longest Track (lngtrk) and Highest-E Shower (heshw)
for comp in ["x", "y", "z"]:
    df[f"res_lngtrk_{comp}"] = np.where(df[f"dir_lngtrk_{comp}"].abs() <= 1.0,
                                        df[f"dir_lngtrk_{comp}"] - df[f"true_dir_{comp}"], np.nan)
    df[f"res_heshw_{comp}"]  = np.where(df[f"dir_heshw_{comp}"].abs() <= 1.0,
                                        df[f"dir_heshw_{comp}"] - df[f"true_dir_{comp}"], np.nan)

# 3D Opening Angle (in degrees) between reconstructed unit vector and true direction
dot_trk = (df["dir_lngtrk_x"] * df["true_dir_x"] +
           df["dir_lngtrk_y"] * df["true_dir_y"] +
           df["dir_lngtrk_z"] * df["true_dir_z"])
dot_shw = (df["dir_heshw_x"] * df["true_dir_x"] +
           df["dir_heshw_y"] * df["true_dir_y"] +
           df["dir_heshw_z"] * df["true_dir_z"])

df["angle_err_lngtrk_deg"] = np.where((df["dir_lngtrk_z"].abs() <= 1.0) & (dot_trk.abs() <= 1.0),
                                      np.degrees(np.arccos(np.clip(dot_trk, -1.0, 1.0))), np.nan)
df["angle_err_heshw_deg"]  = np.where((df["dir_heshw_z"].abs() <= 1.0) & (dot_shw.abs() <= 1.0),
                                      np.degrees(np.arccos(np.clip(dot_shw, -1.0, 1.0))), np.nan)

# ==============================================================================
# HELPER: MULTI-CONDITION DATAFRAME INDEX SELECTION FUNCTION
# ==============================================================================
def select_indices(dataframe, conditions_list, label=""):
    """
    Combines a list of boolean Series with logical AND (&), prints count and
    sample DataFrame indices, and returns the matching pd.Index.
    """
    combined_mask = pd.Series(True, index=dataframe.index)
    for cond in conditions_list:
        combined_mask = combined_mask & cond
    matched_idx = dataframe.index[combined_mask]
    print(f"[{label}] Matched {len(matched_idx)} events | Sample indices: {matched_idx[:8].tolist()}", flush=True)
    return matched_idx

# ==============================================================================
# TOOL 1: True Muons vs True Electrons vs Longest Track vs Highest-E Shower (X, Y, Z)
# ==============================================================================
idx_tool1_numu = select_indices(df, [
    df["reco_ok"] == True,
    df["true_iscc"] == True,
    df["true_pdg"].abs() == 14,
    df["in_fiducial"] == True,
    df["reco_energy_lep_calo"] > 0.5
], label="Tool 1: True nu_mu CC (Fiducial, E_lep > 0.5 GeV)")

idx_tool1_nue = select_indices(df, [
    df["reco_ok"] == True,
    df["true_iscc"] == True,
    df["true_pdg"].abs() == 12,
    df["in_fiducial"] == True,
    df["reco_energy_e_calo"] > 0.5
], label="Tool 1: True nu_e CC (Fiducial, E_ecal > 0.5 GeV)")

# ==============================================================================
# TOOL 2: Residuals vs Longest Track vs Highest-Energy Shower (Multi-Condition Cuts)
# ==============================================================================
idx_tool2_numu_hiE = select_indices(df, [
    df["reco_ok"] == True,
    df["true_iscc"] == True,
    df["true_pdg"].abs() == 14,
    df["reco_energy_lep_calo"] > 1.0,
    df["cvn_numu"] >= 0.5
], label="Tool 2: Selected True nu_mu CC (E_reco > 1 GeV, cvn_numu >= 0.5)")

idx_tool2_nue_hiE = select_indices(df, [
    df["reco_ok"] == True,
    df["true_iscc"] == True,
    df["true_pdg"].abs() == 12,
    df["reco_energy_e_calo"] > 1.0,
    df["cvn_nue"] >= 0.5
], label="Tool 2: Selected True nu_e CC (E_reco > 1 GeV, cvn_nue >= 0.5)")

# ==============================================================================
# TOOL 3: Zenith Angle cos(theta_z) Combined with Energy & Residuals
# ==============================================================================
# Up-going (through Earth: cos_z < -0.2) vs Down-going (from above: cos_z > 0.2), split by Energy > 1 GeV vs <= 1 GeV
idx_tool3_up_hiE = select_indices(df, [
    df["reco_ok"] == True,
    df["true_iscc"] == True,
    df["true_pdg"].abs() == 14,
    df["true_dir_z"] < -0.2,
    df["reco_energy_lep_calo"] > 1.0
], label="Tool 3: Up-going True nu_mu CC (cos_z < -0.2 & E_reco > 1 GeV)")

idx_tool3_down_hiE = select_indices(df, [
    df["reco_ok"] == True,
    df["true_iscc"] == True,
    df["true_pdg"].abs() == 14,
    df["true_dir_z"] > 0.2,
    df["reco_energy_lep_calo"] > 1.0
], label="Tool 3: Down-going True nu_mu CC (cos_z > 0.2 & E_reco > 1 GeV)")

idx_tool3_up_loE = select_indices(df, [
    df["reco_ok"] == True,
    df["true_iscc"] == True,
    df["true_pdg"].abs() == 14,
    df["true_dir_z"] < -0.2,
    df["reco_energy_lep_calo"] <= 1.0,
    df["reco_energy_lep_calo"] > 0.1
], label="Tool 3: Up-going True nu_mu CC (cos_z < -0.2 & 0.1 < E_reco <= 1 GeV)")

# ==============================================================================
# GENERATE 4-PAGE PDF REPORT FOR ALL THREE SELECTION TOOLS
# ==============================================================================
os.makedirs(os.path.expanduser("~/todays_pdfs"), exist_ok=True)
pdf_path = os.path.expanduser("~/todays_pdfs/CAF_Selection_Tools.pdf")

with PdfPages(pdf_path) as pdf:
    # --------------------------------------------------------------------------
    # PAGE 1 (Tool 1): True Muons vs True Electrons vs Longest Track vs Highest-E Shower (X, Y, Z)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    dir_bins = np.linspace(-1.0, 1.0, 51)
    df_t1_mu = df.loc[idx_tool1_numu]
    df_t1_el = df.loc[idx_tool1_nue]

    for i, comp in enumerate(["x", "y", "z"]):
        # Top row: True Muons (nu_mu CC) - True vs lngtrk vs heshw
        axes[0, i].hist(df_t1_mu[f"true_dir_{comp}"].dropna(), bins=dir_bins, histtype="step", lw=2, color="black", label="True Direction")
        axes[0, i].hist(df_t1_mu[f"dir_lngtrk_{comp}"].dropna(), bins=dir_bins, histtype="step", lw=1.8, color="royalblue", label="dir.lngtrk")
        axes[0, i].hist(df_t1_mu[f"dir_heshw_{comp}"].dropna(), bins=dir_bins, histtype="step", lw=1.8, ls="--", color="darkorange", label="dir.heshw")
        axes[0, i].set_title(f"Tool 1 (True $\\nu_\\mu$ CC): Direction {comp.upper()}")
        axes[0, i].set_xlabel(f"Unit Direction {comp}")
        axes[0, i].set_ylabel("Events")
        axes[0, i].legend(fontsize=8)
        axes[0, i].grid(alpha=0.3)

        # Bottom row: True Electrons (nu_e CC) - True vs lngtrk vs heshw
        axes[1, i].hist(df_t1_el[f"true_dir_{comp}"].dropna(), bins=dir_bins, histtype="step", lw=2, color="black", label="True Direction")
        axes[1, i].hist(df_t1_el[f"dir_lngtrk_{comp}"].dropna(), bins=dir_bins, histtype="step", lw=1.8, ls="--", color="royalblue", label="dir.lngtrk")
        axes[1, i].hist(df_t1_el[f"dir_heshw_{comp}"].dropna(), bins=dir_bins, histtype="step", lw=1.8, color="crimson", label="dir.heshw")
        axes[1, i].set_title(f"Tool 1 (True $\\nu_e$ CC): Direction {comp.upper()}")
        axes[1, i].set_xlabel(f"Unit Direction {comp}")
        axes[1, i].set_ylabel("Events")
        axes[1, i].legend(fontsize=8)
        axes[1, i].grid(alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # PAGE 2 (Tool 1b): Component Residuals (Reco - True) for X, Y, Z (Track vs Shower)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    res_d_bins = np.linspace(-1.0, 1.0, 61)

    for i, comp in enumerate(["x", "y", "z"]):
        axes[0, i].hist(df_t1_mu[f"res_lngtrk_{comp}"].dropna(), bins=res_d_bins, histtype="step", lw=2, color="royalblue", label="lngtrk - true")
        axes[0, i].hist(df_t1_mu[f"res_heshw_{comp}"].dropna(), bins=res_d_bins, histtype="step", lw=2, ls="--", color="darkorange", label="heshw - true")
        axes[0, i].axvline(0, color="k", ls=":", lw=1)
        axes[0, i].set_title(f"True $\\nu_\\mu$ CC: $\\Delta$dir.{comp} Residual")
        axes[0, i].set_xlabel(f"Reco {comp} - True {comp}")
        axes[0, i].legend(fontsize=8)
        axes[0, i].grid(alpha=0.3)

        axes[1, i].hist(df_t1_el[f"res_lngtrk_{comp}"].dropna(), bins=res_d_bins, histtype="step", lw=2, ls="--", color="royalblue", label="lngtrk - true")
        axes[1, i].hist(df_t1_el[f"res_heshw_{comp}"].dropna(), bins=res_d_bins, histtype="step", lw=2, color="crimson", label="heshw - true")
        axes[1, i].axvline(0, color="k", ls=":", lw=1)
        axes[1, i].set_title(f"True $\\nu_e$ CC: $\\Delta$dir.{comp} Residual")
        axes[1, i].set_xlabel(f"Reco {comp} - True {comp}")
        axes[1, i].legend(fontsize=8)
        axes[1, i].grid(alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # PAGE 3 (Tool 2): Residuals vs Longest Track vs Highest-Energy Shower (E_reco > 1 GeV & CVN >= 0.5)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    df_t2_mu = df.loc[idx_tool2_numu_hiE]
    df_t2_el = df.loc[idx_tool2_nue_hiE]
    ang_bins = np.linspace(0, 90, 61)

    # Left: 3D Opening Angle Residual (deg) for lngtrk vs heshw on selected high-E events
    axes[0].hist(df_t2_mu["angle_err_lngtrk_deg"].dropna(), bins=ang_bins, density=True, histtype="step", lw=2, color="royalblue", label=r"Selected $\nu_\mu$ CC: lngtrk (Track)")
    axes[0].hist(df_t2_mu["angle_err_heshw_deg"].dropna(), bins=ang_bins, density=True, histtype="step", lw=1.8, ls="--", color="cornflowerblue", label=r"Selected $\nu_\mu$ CC: heshw (Shower)")
    axes[0].hist(df_t2_el["angle_err_heshw_deg"].dropna(), bins=ang_bins, density=True, histtype="step", lw=2, color="crimson", label=r"Selected $\nu_e$ CC: heshw (Shower)")
    axes[0].hist(df_t2_el["angle_err_lngtrk_deg"].dropna(), bins=ang_bins, density=True, histtype="step", lw=1.8, ls="--", color="salmon", label=r"Selected $\nu_e$ CC: lngtrk (Track)")
    axes[0].set_title(r"Tool 2: 3D Opening Angle Residual ($E_{\mathrm{reco}}>1$ GeV, CVN$\geq 0.5$)")
    axes[0].set_xlabel(r"3D Opening Angle $\Delta\theta_{\mathrm{reco,true}}$ [Degrees]")
    axes[0].set_ylabel("Normalized Events")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Right: Energy Residual for well-reconstructed track angle (< 15 deg) vs poorly reconstructed (> 30 deg)
    well_trk = df_t2_mu[df_t2_mu["angle_err_lngtrk_deg"] < 15.0]["frac_res_lep_calo"].dropna()
    poor_trk = df_t2_mu[df_t2_mu["angle_err_lngtrk_deg"] > 30.0]["frac_res_lep_calo"].dropna()
    well_shw = df_t2_el[df_t2_el["angle_err_heshw_deg"] < 15.0]["frac_res_e_calo"].dropna()

    ebins_res = np.linspace(-1.0, 1.0, 61)
    axes[1].hist(well_trk, bins=ebins_res, density=True, histtype="step", lw=2, color="royalblue", label=r"$\nu_\mu$ CC: Good Track ($\Delta\theta_{\mathrm{trk}} < 15^\circ$)")
    axes[1].hist(poor_trk, bins=ebins_res, density=True, histtype="step", lw=2, ls="--", color="navy", label=r"$\nu_\mu$ CC: Poor Track ($\Delta\theta_{\mathrm{trk}} > 30^\circ$)")
    axes[1].hist(well_shw, bins=ebins_res, density=True, histtype="step", lw=2, color="crimson", label=r"$\nu_e$ CC: Good Shower ($\Delta\theta_{\mathrm{shw}} < 15^\circ$)")
    axes[1].axvline(0, color="k", ls=":", lw=1)
    axes[1].set_title("Tool 2: Energy Residuals Split by Track/Shower Angle Quality")
    axes[1].set_xlabel(r"Fractional Energy Residual $(E_{\mathrm{reco}} - E_{\mathrm{true}}) / E_{\mathrm{true}}$")
    axes[1].set_ylabel("Normalized Events")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # PAGE 4 (Tool 3): Zenith Angle cos(theta_z) Combined with Energy & Residuals
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    df_t3_up_hi   = df.loc[idx_tool3_up_hiE]
    df_t3_down_hi = df.loc[idx_tool3_down_hiE]
    df_t3_up_lo   = df.loc[idx_tool3_up_loE]

    # Left: Energy Residuals across Zenith Angle & Energy regimes
    axes[0].hist(df_t3_up_hi["frac_res_lep_calo"].dropna(), bins=ebins_res, density=True, histtype="step", lw=2, color="royalblue", label=r"Up-going ($\cos\theta_z < -0.2$) & $E_{\mathrm{reco}} > 1$ GeV")
    axes[0].hist(df_t3_down_hi["frac_res_lep_calo"].dropna(), bins=ebins_res, density=True, histtype="step", lw=2, ls="--", color="darkgreen", label=r"Down-going ($\cos\theta_z > 0.2$) & $E_{\mathrm{reco}} > 1$ GeV")
    axes[0].hist(df_t3_up_lo["frac_res_lep_calo"].dropna(), bins=ebins_res, density=True, histtype="step", lw=2, color="darkorange", label=r"Up-going ($\cos\theta_z < -0.2$) & $0.1 < E_{\mathrm{reco}} \leq 1$ GeV")
    axes[0].axvline(0, color="k", ls=":", lw=1)
    axes[0].set_title(r"Tool 3: Energy Residuals by Zenith Angle & Energy Cuts")
    axes[0].set_xlabel(r"Fractional Energy Residual $(E_{\mathrm{reco}} - E_{\mathrm{true}}) / E_{\mathrm{true}}$")
    axes[0].set_ylabel("Normalized Events")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Right: Zenith Angle Residual (cos_z_reco - cos_z_true) for High-E vs Low-E
    z_res_bins = np.linspace(-1.0, 1.0, 61)
    axes[1].hist(df_t3_up_hi["res_lngtrk_z"].dropna(), bins=z_res_bins, density=True, histtype="step", lw=2, color="royalblue", label=r"Up-going ($\cos\theta_z < -0.2$) & $E_{\mathrm{reco}} > 1$ GeV")
    axes[1].hist(df_t3_down_hi["res_lngtrk_z"].dropna(), bins=z_res_bins, density=True, histtype="step", lw=2, ls="--", color="darkgreen", label=r"Down-going ($\cos\theta_z > 0.2$) & $E_{\mathrm{reco}} > 1$ GeV")
    axes[1].hist(df_t3_up_lo["res_lngtrk_z"].dropna(), bins=z_res_bins, density=True, histtype="step", lw=2, color="darkorange", label=r"Up-going ($\cos\theta_z < -0.2$) & $0.1 < E_{\mathrm{reco}} \leq 1$ GeV")
    axes[1].axvline(0, color="k", ls=":", lw=1)
    axes[1].set_title(r"Tool 3: Zenith Angle ($\cos\theta_z$) Residuals by Energy Regime")
    axes[1].set_xlabel(r"Zenith Angle Residual $\cos\theta_{z,\mathrm{reco}} - \cos\theta_{z,\mathrm{true}}$")
    axes[1].set_ylabel("Normalized Events")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"SUCCESS! Saved 4-page Selection Tools PDF to: {pdf_path}", flush=True)