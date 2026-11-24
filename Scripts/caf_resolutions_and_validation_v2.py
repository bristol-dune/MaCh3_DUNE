#!/usr/bin/env python3
import os
import sys
import math
from array import array
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadTickX(1)
ROOT.gStyle.SetPadTickY(1)

for p in ["./build/lib", "../build/lib", "./build", "../build"]:
    ROOT.gSystem.AddDynamicPath(p)
ROOT.gSystem.Load("libduneanaobj_StandardRecord")

caf_path = "/storage/1/st15719/dune/atmo/Mach3_Inputs/CAFs/caf_new_sum.2.6M_weighted.root"
print("Opening CAF file:", caf_path, flush=True)
f = ROOT.TFile.Open(caf_path, "READ")
caf_tree = f.Get("cafTree")

# Speed up reading 20x & silence Near Detector (nd.sand) warnings

max_events = 250000 if len(sys.argv) <= 1 else (-1 if sys.argv[1].lower() == "all" else int(sys.argv[1]))

# Energy bins for Resolution vs. True Energy
e_bins = [0.2, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0, 7.5, 10.0]
n_ebins = len(e_bins) - 1

# ==============================================================================
# 1. BOOK HISTOGRAMS
# ==============================================================================
# Page 1: Energy Residuals (lep_calo vs e_calo) on True nu_mu CC vs True nu_e CC (True Electrons)
h_res_lep_numu  = ROOT.TH1D("h_res_lep_numu", "Energy Residuals: True #nu_{#mu} CC (|PDG|=14, CC);(E_{reco} - E_{true}) / E_{true};Events", 80, -1.0, 1.0)
h_res_ecal_numu = ROOT.TH1D("h_res_ecal_numu", "Energy Residuals: True #nu_{#mu} CC (|PDG|=14, CC);(E_{reco} - E_{true}) / E_{true};Events", 80, -1.0, 1.0)

h_res_lep_nue   = ROOT.TH1D("h_res_lep_nue", "Energy Residuals: True #nu_{e} CC / Electrons (|PDG|=12, CC);(E_{reco} - E_{true}) / E_{true};Events", 80, -1.0, 1.0)
h_res_ecal_nue  = ROOT.TH1D("h_res_ecal_nue", "Energy Residuals: True #nu_{e} CC / Electrons (|PDG|=12, CC);(E_{reco} - E_{true}) / E_{true};Events", 80, -1.0, 1.0)

# Page 2 Left: Muon Energy Versions (All True nu_mu CC vs Fiducial Volume + Contained Selection)
h_res_rng_all = ROOT.TH1D("h_res_rng_all", "Muon Energy Residuals: True #nu_{#mu} CC (Fiducial vs All);(E_{reco} - E_{true}) / E_{true};Events", 80, -1.0, 1.0)
h_res_rng_fid = ROOT.TH1D("h_res_rng_fid", "Muon Energy Residuals: True #nu_{#mu} CC (Fiducial vs All);(E_{reco} - E_{true}) / E_{true};Events", 80, -1.0, 1.0)
h_res_mcs_fid = ROOT.TH1D("h_res_mcs_fid", "Muon Energy Residuals: True #nu_{#mu} CC (Fiducial vs All);(E_{reco} - E_{true}) / E_{true};Events", 80, -1.0, 1.0)
h_res_lep_fid = ROOT.TH1D("h_res_lep_fid", "Muon Energy Residuals: True #nu_{#mu} CC (Fiducial vs All);(E_{reco} - E_{true}) / E_{true};Events", 80, -1.0, 1.0)

# Page 2 Right: Direction Residuals (lngtrk vs heshw) on True nu_mu CC AND True nu_e CC
h_dir_trk_numu = ROOT.TH1D("h_dir_trk_numu", "Direction (cos#theta_{z}) Residuals: True #nu_{#mu} CC vs True #nu_{e} CC;cos#theta_{z,reco} - cos#theta_{z,true};Normalized Shape", 80, -1.0, 1.0)
h_dir_shw_numu = ROOT.TH1D("h_dir_shw_numu", "Direction (cos#theta_{z}) Residuals: True #nu_{#mu} CC vs True #nu_{e} CC;cos#theta_{z,reco} - cos#theta_{z,true};Normalized Shape", 80, -1.0, 1.0)
h_dir_trk_nue  = ROOT.TH1D("h_dir_trk_nue",  "Direction (cos#theta_{z}) Residuals: True #nu_{#mu} CC vs True #nu_{e} CC;cos#theta_{z,reco} - cos#theta_{z,true};Normalized Shape", 80, -1.0, 1.0)
h_dir_shw_nue  = ROOT.TH1D("h_dir_shw_nue",  "Direction (cos#theta_{z}) Residuals: True #nu_{#mu} CC vs True #nu_{e} CC;cos#theta_{z,reco} - cos#theta_{z,true};Normalized Shape", 80, -1.0, 1.0)

# Page 4 Left: Valid CVN score distributions (0 <= cvn <= 1) Normalized Shape
h_cvn_numu_sig = ROOT.TH1D("h_cvn_sig", "CVN #nu_{#mu} Classifier Score (Valid Scores 0#leqcvn#leq1);cvn.numu Score;Fraction of Events (Normalized)", 50, 0.0, 1.0001)
h_cvn_numu_nue = ROOT.TH1D("h_cvn_nue", "CVN #nu_{#mu} Classifier Score (Valid Scores 0#leqcvn#leq1);cvn.numu Score;Fraction of Events (Normalized)", 50, 0.0, 1.0001)
h_cvn_numu_nc  = ROOT.TH1D("h_cvn_nc",  "CVN #nu_{#mu} Classifier Score (Valid Scores 0#leqcvn#leq1);cvn.numu Score;Fraction of Events (Normalized)", 50, 0.0, 1.0001)

# Page 5 Top: Tightened Vertex X, Y, Z (Z tightened to [-100, 1500] cm) + Fiducial Cut
h_vtx_x_all = ROOT.TH1D("h_vtx_x_all", "True Vertex X (Tightened);mc.nu.vtx.x [cm];Events", 70, -420, 420)
h_vtx_x_fid = ROOT.TH1D("h_vtx_x_fid", "True Vertex X (Tightened);mc.nu.vtx.x [cm];Events", 70, -420, 420)
h_vtx_y_all = ROOT.TH1D("h_vtx_y_all", "True Vertex Y (Tightened);mc.nu.vtx.y [cm];Events", 70, -680, 680)
h_vtx_y_fid = ROOT.TH1D("h_vtx_y_fid", "True Vertex Y (Tightened);mc.nu.vtx.y [cm];Events", 70, -680, 680)
h_vtx_z_all = ROOT.TH1D("h_vtx_z_all", "True Vertex Z (Tightened to Detector);mc.nu.vtx.z [cm];Events", 80, -100, 1500)
h_vtx_z_fid = ROOT.TH1D("h_vtx_z_fid", "True Vertex Z (Tightened to Detector);mc.nu.vtx.z [cm];Events", 80, -100, 1500)

# Page 5 Bottom: Normalized Momentum -> Unit Direction Cosines (p_i / |p|)
h_udir_x_all  = ROOT.TH1D("h_udir_x_all",  "Normalized Direction p_{x} / |p|;p_{x} / |p|;Events", 60, -1.0, 1.0)
h_udir_x_numu = ROOT.TH1D("h_udir_x_numu", "Normalized Direction p_{x} / |p|;p_{x} / |p|;Events", 60, -1.0, 1.0)
h_udir_y_all  = ROOT.TH1D("h_udir_y_all",  "Normalized Direction p_{y} / |p|;p_{y} / |p|;Events", 60, -1.0, 1.0)
h_udir_y_numu = ROOT.TH1D("h_udir_y_numu", "Normalized Direction p_{y} / |p|;p_{y} / |p|;Events", 60, -1.0, 1.0)
h_udir_z_all  = ROOT.TH1D("h_udir_z_all",  "Normalized Direction p_{z} / |p|;p_{z} / |p|;Events", 60, -1.0, 1.0)
h_udir_z_numu = ROOT.TH1D("h_udir_z_numu", "Normalized Direction p_{z} / |p|;p_{z} / |p|;Events", 60, -1.0, 1.0)

# Lists for Page 3 Binned Energy Resolution (True nu_mu CC and True nu_e CC)
res_bins_lep_numu  = [[] for _ in range(n_ebins)]
res_bins_ecal_numu = [[] for _ in range(n_ebins)]
res_bins_rng_numu  = [[] for _ in range(n_ebins)]
res_bins_mcs_numu  = [[] for _ in range(n_ebins)]
res_bins_ecal_nue  = [[] for _ in range(n_ebins)]

# ROC Curve arrays (100 thresholds from 0.00 to 0.99)
n_cuts = 100
thresholds = [i / float(n_cuts) for i in range(n_cuts)]
sig_pass_cuts = [0] * n_cuts
bkg_pass_cuts = [0] * n_cuts
total_valid_sig = 0
total_valid_bkg = 0

def find_ebin(e_val):
    for i in range(n_ebins):
        if e_bins[i] <= e_val < e_bins[i+1]:
            return i
    return -1

def is_in_fiducial(vx, vy, vz):
    # Trim 50 cm inside active TPC walls: |x| < 310 cm, |y| < 550 cm, 50 < z < 1350 cm
    return (abs(vx) < 310.0) and (abs(vy) < 550.0) and (50.0 < vz < 1350.0)

print("Looping over cafTree events...", flush=True)
for n_proc, entry in enumerate(caf_tree):
    if 0 < max_events <= n_proc:
        break
    if entry.rec.mc.nu.size() != 1:
        continue

    nu = entry.rec.mc.nu[0]
    true_pdg  = int(nu.pdg)
    true_e    = float(nu.E)
    true_iscc = bool(nu.iscc)
    px, py, pz = float(nu.momentum.x), float(nu.momentum.y), float(nu.momentum.z)
    vx, vy, vz = float(nu.vtx.x), float(nu.vtx.y), float(nu.vtx.z)
    p_mag = math.sqrt(px*px + py*py + pz*pz)

    if entry.rec.common.ixn.pandora.size() != 1:
        continue

    r = entry.rec.common.ixn.pandora[0]
    e_lep  = float(r.Enu.lep_calo)
    e_ecal = float(r.Enu.e_calo)
    e_rng  = float(r.Enu.mu_range)
    e_mcs  = float(r.Enu.mu_mcs)

    dir_trk_z = float(r.dir.lngtrk.z)
    dir_shw_z = float(r.dir.heshw.z)
    cvn_numu  = float(r.nuhyp.cvn.numu)

    # 1. Vertex & Fiducial Volume checks
    in_fid = is_in_fiducial(vx, vy, vz)
    h_vtx_x_all.Fill(vx); h_vtx_y_all.Fill(vy); h_vtx_z_all.Fill(vz)
    if in_fid:
        h_vtx_x_fid.Fill(vx); h_vtx_y_fid.Fill(vy); h_vtx_z_fid.Fill(vz)

    # 2. Normalized Momentum Direction Cosines (p_i / |p|)
    if p_mag > 0:
        ux, uy, uz = px / p_mag, py / p_mag, pz / p_mag
        h_udir_x_all.Fill(ux); h_udir_y_all.Fill(uy); h_udir_z_all.Fill(uz)
    else:
        ux, uy, uz = -999.0, -999.0, -999.0

    # Residuals
    res_lep  = (e_lep - true_e) / true_e if (e_lep > 0 and true_e > 0) else None
    res_ecal = (e_ecal - true_e) / true_e if (e_ecal > 0 and true_e > 0) else None
    res_rng  = (e_rng - true_e) / true_e if (e_rng > 0 and true_e > 0) else None
    res_mcs  = (e_mcs - true_e) / true_e if (e_mcs > 0 and true_e > 0) else None

    is_true_numu_cc = (abs(true_pdg) == 14 and true_iscc)
    is_true_nue_cc  = (abs(true_pdg) == 12 and true_iscc)

    # 3. CVN & ROC Curve accumulation (only events where CVN evaluated a valid score in [0, 1])
    if 0.0 <= cvn_numu <= 1.0:
        if not true_iscc:
            h_cvn_numu_nc.Fill(cvn_numu)
        elif is_true_numu_cc:
            h_cvn_numu_sig.Fill(cvn_numu)
        elif is_true_nue_cc:
            h_cvn_numu_nue.Fill(cvn_numu)

        if is_true_numu_cc:
            total_valid_sig += 1
            max_idx = min(n_cuts - 1, int(cvn_numu * n_cuts))
            for idx in range(max_idx + 1):
                sig_pass_cuts[idx] += 1
        else:
            total_valid_bkg += 1
            max_idx = min(n_cuts - 1, int(cvn_numu * n_cuts))
            for idx in range(max_idx + 1):
                bkg_pass_cuts[idx] += 1

    # 4. True Muon Neutrino CC Plots
    eb_idx = find_ebin(true_e)
    if is_true_numu_cc:
        if p_mag > 0:
            h_udir_x_numu.Fill(ux); h_udir_y_numu.Fill(uy); h_udir_z_numu.Fill(uz)
        if res_lep is not None:  h_res_lep_numu.Fill(res_lep)
        if res_ecal is not None: h_res_ecal_numu.Fill(res_ecal)
        if res_rng is not None:  h_res_rng_all.Fill(res_rng)

        # Fiducial + CVN selected nu_mu CC for cleaner containment comparison
        if in_fid and cvn_numu >= 0.5:
            if res_rng is not None: h_res_rng_fid.Fill(res_rng)
            if res_mcs is not None: h_res_mcs_fid.Fill(res_mcs)
            if res_lep is not None: h_res_lep_fid.Fill(res_lep)

        if dir_trk_z > -990 and uz > -990:
            h_dir_trk_numu.Fill(dir_trk_z - uz)
        if dir_shw_z > -990 and uz > -990:
            h_dir_shw_numu.Fill(dir_shw_z - uz)

        if eb_idx >= 0:
            if res_lep is not None and -1.0 <= res_lep <= 1.5:   res_bins_lep_numu[eb_idx].append(res_lep)
            if res_ecal is not None and -1.0 <= res_ecal <= 1.5: res_bins_ecal_numu[eb_idx].append(res_ecal)
            if res_rng is not None and -1.0 <= res_rng <= 1.5:   res_bins_rng_numu[eb_idx].append(res_rng)
            if res_mcs is not None and -1.0 <= res_mcs <= 1.5:   res_bins_mcs_numu[eb_idx].append(res_mcs)

    # 5. True Electron Neutrino CC Plots (True Electrons)
    if is_true_nue_cc:
        if res_lep is not None:  h_res_lep_nue.Fill(res_lep)
        if res_ecal is not None: h_res_ecal_nue.Fill(res_ecal)
        if dir_trk_z > -990 and uz > -990:
            h_dir_trk_nue.Fill(dir_trk_z - uz)
        if dir_shw_z > -990 and uz > -990:
            h_dir_shw_nue.Fill(dir_shw_z - uz)
        if eb_idx >= 0 and res_ecal is not None and -1.0 <= res_ecal <= 1.5:
            res_bins_ecal_nue[eb_idx].append(res_ecal)

    if (n_proc + 1) % 50000 == 0:
        print(f"  Processed {n_proc + 1} events...", flush=True)

# Helper to build TGraphErrors for Resolution (68% half-width) & Median Bias
def make_res_and_bias_graphs(binned_lists, color, marker, style=1):
    g_res  = ROOT.TGraphErrors()
    g_bias = ROOT.TGraphErrors()
    pt = 0
    for i in range(n_ebins):
        arr = sorted(binned_lists[i])
        n = len(arr)
        if n < 20:
            continue
        x_center = 0.5 * (e_bins[i] + e_bins[i+1])
        x_err    = 0.5 * (e_bins[i+1] - e_bins[i])
        q16 = arr[int(0.16 * n)]
        q50 = arr[int(0.50 * n)]
        q84 = arr[int(0.84 * n)]
        iqr68 = 0.5 * (q84 - q16)
        err   = iqr68 / math.sqrt(2.0 * n)

        g_res.SetPoint(pt, x_center, iqr68)
        g_res.SetPointError(pt, x_err, err)
        g_bias.SetPoint(pt, x_center, q50)
        g_bias.SetPointError(pt, x_err, err)
        pt += 1

    for g in (g_res, g_bias):
        g.SetLineColor(color); g.SetMarkerColor(color)
        g.SetMarkerStyle(marker); g.SetLineWidth(2); g.SetLineStyle(style)
    return g_res, g_bias

g_res_lep_numu,  g_bias_lep_numu  = make_res_and_bias_graphs(res_bins_lep_numu,  ROOT.kBlue+1,   20, 1)
g_res_ecal_numu, g_bias_ecal_numu = make_res_and_bias_graphs(res_bins_ecal_numu, ROOT.kRed+1,    21, 1)
g_res_rng_numu,  g_bias_rng_numu  = make_res_and_bias_graphs(res_bins_rng_numu,  ROOT.kGreen+2,  22, 2)
g_res_mcs_numu,  g_bias_mcs_numu  = make_res_and_bias_graphs(res_bins_mcs_numu,  ROOT.kOrange+7, 23, 2)
g_res_ecal_nue,  g_bias_ecal_nue  = make_res_and_bias_graphs(res_bins_ecal_nue,  ROOT.kMagenta+2, 29, 1)

def style_h(h, color, ls=1):
    h.SetLineColor(color); h.SetLineWidth(2); h.SetLineStyle(ls)

def norm_h(h):
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())

os.makedirs(os.path.expanduser("~/todays_pdfs"), exist_ok=True)
pdf_path = os.path.expanduser("~/todays_pdfs/CAF_Resolutions_and_Validation_v2.pdf")

c = ROOT.TCanvas("c", "CAF Analysis v2", 1200, 550)
c.Print(pdf_path + "[")

# ==============================================================================
# PAGE 1: Energy Residuals (lep_calo vs e_calo) on True nu_mu CC vs True nu_e CC
# ==============================================================================
c.Clear(); c.Divide(2, 1)
c.cd(1); ROOT.gPad.SetGrid()
style_h(h_res_lep_numu, ROOT.kBlue+1); style_h(h_res_ecal_numu, ROOT.kRed+1)
h_res_lep_numu.SetMaximum(max(h_res_lep_numu.GetMaximum(), h_res_ecal_numu.GetMaximum()) * 1.15)
h_res_lep_numu.Draw("HIST"); h_res_ecal_numu.Draw("HIST SAME")
leg1 = ROOT.TLegend(0.48, 0.72, 0.88, 0.88)
leg1.AddEntry(h_res_lep_numu, "lep_calo on True #nu_{#mu} CC", "l")
leg1.AddEntry(h_res_ecal_numu, "e_calo on True #nu_{#mu} CC", "l")
leg1.Draw()

c.cd(2); ROOT.gPad.SetGrid()
style_h(h_res_ecal_nue, ROOT.kRed+1); style_h(h_res_lep_nue, ROOT.kBlue+1)
h_res_ecal_nue.SetMaximum(max(h_res_ecal_nue.GetMaximum(), h_res_lep_nue.GetMaximum()) * 1.15)
h_res_ecal_nue.Draw("HIST"); h_res_lep_nue.Draw("HIST SAME")
leg2 = ROOT.TLegend(0.48, 0.72, 0.88, 0.88)
leg2.AddEntry(h_res_ecal_nue, "e_calo on True #nu_{e} CC (Electrons)", "l")
leg2.AddEntry(h_res_lep_nue, "lep_calo on True #nu_{e} CC", "l")
leg2.Draw()
c.Print(pdf_path)

# ==============================================================================
# PAGE 2: Left = Muon Energy Versions (Fiducial+Selected vs Uncut); Right = Direction Residuals (nu_mu vs nu_e)
# ==============================================================================
c.Clear(); c.Divide(2, 1)
c.cd(1); ROOT.gPad.SetGrid()
style_h(h_res_rng_all, ROOT.kGray+2, 2)
style_h(h_res_rng_fid, ROOT.kGreen+2, 1)
style_h(h_res_mcs_fid, ROOT.kOrange+7, 1)
style_h(h_res_lep_fid, ROOT.kBlue+1, 1)
h_res_rng_all.SetMaximum(max(h_res_rng_all.GetMaximum(), h_res_rng_fid.GetMaximum()) * 1.15)
h_res_rng_all.Draw("HIST"); h_res_rng_fid.Draw("HIST SAME"); h_res_mcs_fid.Draw("HIST SAME"); h_res_lep_fid.Draw("HIST SAME")
leg3 = ROOT.TLegend(0.42, 0.65, 0.88, 0.88)
leg3.AddEntry(h_res_rng_all, "mu_range (All True #nu_{#mu} CC - No Cuts)", "l")
leg3.AddEntry(h_res_rng_fid, "mu_range (Fiducial + cvn#geq0.5)", "l")
leg3.AddEntry(h_res_mcs_fid, "mu_mcs (Fiducial + cvn#geq0.5)", "l")
leg3.AddEntry(h_res_lep_fid, "lep_calo (Fiducial + cvn#geq0.5)", "l")
leg3.Draw()

c.cd(2); ROOT.gPad.SetGrid()
for h in (h_dir_trk_numu, h_dir_shw_numu, h_dir_trk_nue, h_dir_shw_nue):
    norm_h(h)
style_h(h_dir_trk_numu, ROOT.kBlue+1, 1)
style_h(h_dir_shw_numu, ROOT.kBlue+1, 2)
style_h(h_dir_shw_nue,  ROOT.kRed+1, 1)
style_h(h_dir_trk_nue,  ROOT.kRed+1, 2)
ymax = max(h_dir_trk_numu.GetMaximum(), h_dir_shw_nue.GetMaximum()) * 1.18
h_dir_trk_numu.SetMaximum(ymax)
h_dir_trk_numu.Draw("HIST"); h_dir_shw_numu.Draw("HIST SAME"); h_dir_shw_nue.Draw("HIST SAME"); h_dir_trk_nue.Draw("HIST SAME")
leg4 = ROOT.TLegend(0.42, 0.65, 0.88, 0.88)
leg4.AddEntry(h_dir_trk_numu, "True #nu_{#mu} CC: dir.lngtrk.z (Track)", "l")
leg4.AddEntry(h_dir_shw_numu, "True #nu_{#mu} CC: dir.heshw.z (Shower)", "l")
leg4.AddEntry(h_dir_shw_nue,  "True #nu_{e} CC: dir.heshw.z (Shower)", "l")
leg4.AddEntry(h_dir_trk_nue,  "True #nu_{e} CC: dir.lngtrk.z (Track)", "l")
leg4.Draw()
c.Print(pdf_path)

# ==============================================================================
# PAGE 3: Energy Resolution (68% Width) & Bias vs True Energy (with True nu_e e_calo added)
# ==============================================================================
c.Clear(); c.Divide(2, 1)
c.cd(1); ROOT.gPad.SetGrid()
frame_res = ROOT.TH1D("frame_res", "Energy Resolution (68% Quantile Width) vs. E_{true};True Neutrino Energy E_{true} [GeV];Energy Resolution #sigma_{E} / E_{true} (68% Width)", 10, 0.0, 10.0)
frame_res.SetMinimum(0.0); frame_res.SetMaximum(0.65); frame_res.Draw()
g_res_lep_numu.Draw("LP SAME"); g_res_ecal_numu.Draw("LP SAME")
g_res_rng_numu.Draw("LP SAME"); g_res_mcs_numu.Draw("LP SAME"); g_res_ecal_nue.Draw("LP SAME")
leg5 = ROOT.TLegend(0.45, 0.62, 0.88, 0.88)
leg5.AddEntry(g_res_lep_numu,  "True #nu_{#mu} CC: lep_calo", "lp")
leg5.AddEntry(g_res_ecal_numu, "True #nu_{#mu} CC: e_calo", "lp")
leg5.AddEntry(g_res_rng_numu,  "True #nu_{#mu} CC: mu_range", "lp")
leg5.AddEntry(g_res_mcs_numu,  "True #nu_{#mu} CC: mu_mcs", "lp")
leg5.AddEntry(g_res_ecal_nue,  "True #nu_{e} CC: e_calo", "lp")
leg5.Draw()

c.cd(2); ROOT.gPad.SetGrid()
frame_bias = ROOT.TH1D("frame_bias", "Median Energy Scale Bias vs. E_{true};True Neutrino Energy E_{true} [GeV];Median (E_{reco} - E_{true}) / E_{true}", 10, 0.0, 10.0)
frame_bias.SetMinimum(-0.6); frame_bias.SetMaximum(0.4); frame_bias.Draw()
g_bias_lep_numu.Draw("LP SAME"); g_bias_ecal_numu.Draw("LP SAME")
g_bias_rng_numu.Draw("LP SAME"); g_bias_mcs_numu.Draw("LP SAME"); g_bias_ecal_nue.Draw("LP SAME")
leg6 = ROOT.TLegend(0.45, 0.62, 0.88, 0.88)
leg6.AddEntry(g_bias_lep_numu,  "True #nu_{#mu} CC: lep_calo", "lp")
leg6.AddEntry(g_bias_ecal_numu, "True #nu_{#mu} CC: e_calo", "lp")
leg6.AddEntry(g_bias_rng_numu,  "True #nu_{#mu} CC: mu_range", "lp")
leg6.AddEntry(g_bias_mcs_numu,  "True #nu_{#mu} CC: mu_mcs", "lp")
leg6.AddEntry(g_bias_ecal_nue,  "True #nu_{e} CC: e_calo", "lp")
leg6.Draw()
c.Print(pdf_path)

# ==============================================================================
# PAGE 4: Left = Normalized Valid CVN Score (Linear Scale); Right = ROC Curve & Sweet Spot
# ==============================================================================
c.Clear(); c.Divide(2, 1)
c.cd(1); ROOT.gPad.SetGrid(); ROOT.gPad.SetLogy(0)
for h in (h_cvn_numu_sig, h_cvn_numu_nue, h_cvn_numu_nc):
    norm_h(h)
style_h(h_cvn_numu_sig, ROOT.kBlue+1); style_h(h_cvn_numu_nue, ROOT.kRed+1); style_h(h_cvn_numu_nc, ROOT.kGreen+2)
ymax_cvn = max(h_cvn_numu_sig.GetMaximum(), h_cvn_numu_nue.GetMaximum(), h_cvn_numu_nc.GetMaximum()) * 1.15
h_cvn_numu_sig.SetMaximum(ymax_cvn)
h_cvn_numu_sig.Draw("HIST"); h_cvn_numu_nue.Draw("HIST SAME"); h_cvn_numu_nc.Draw("HIST SAME")
leg7 = ROOT.TLegend(0.30, 0.68, 0.85, 0.88)
leg7.AddEntry(h_cvn_numu_sig, "True #nu_{#mu} CC Signal (Peaks at 1.0)", "l")
leg7.AddEntry(h_cvn_numu_nue, "True #nu_{e} CC Background (Peaks at 0.0)", "l")
leg7.AddEntry(h_cvn_numu_nc,  "True NC Background (Peaks at 0.0)", "l")
leg7.Draw()

c.cd(2); ROOT.gPad.SetGrid(); ROOT.gPad.SetLogy(0)
g_roc = ROOT.TGraph()
g_eff_pur = ROOT.TGraph()
g_sweet_roc = ROOT.TGraph()
g_sweet_ep  = ROOT.TGraph()

best_score = -1.0
best_cut = 0.5
best_tpr, best_fpr, best_pur = 0.0, 0.0, 0.0

for idx in range(n_cuts):
    tpr = sig_pass_cuts[idx] / float(total_valid_sig) if total_valid_sig > 0 else 0.0
    fpr = bkg_pass_cuts[idx] / float(total_valid_bkg) if total_valid_bkg > 0 else 0.0
    tot_pass = sig_pass_cuts[idx] + bkg_pass_cuts[idx]
    pur = sig_pass_cuts[idx] / float(tot_pass) if tot_pass > 0 else 0.0

    g_roc.SetPoint(idx, fpr, tpr)
    g_eff_pur.SetPoint(idx, tpr, pur)

    # Sweet spot maximizes Efficiency * Purity
    if (tpr * pur) > best_score:
        best_score = tpr * pur
        best_cut = thresholds[idx]
        best_tpr, best_fpr, best_pur = tpr, fpr, pur

g_roc.SetLineColor(ROOT.kBlue+1); g_roc.SetLineWidth(3)
g_eff_pur.SetLineColor(ROOT.kOrange+7); g_eff_pur.SetLineWidth(3); g_eff_pur.SetLineStyle(2)

g_sweet_roc.SetPoint(0, best_fpr, best_tpr)
g_sweet_roc.SetMarkerStyle(29); g_sweet_roc.SetMarkerSize(2.2); g_sweet_roc.SetMarkerColor(ROOT.kRed+1)
g_sweet_ep.SetPoint(0, best_tpr, best_pur)
g_sweet_ep.SetMarkerStyle(20); g_sweet_ep.SetMarkerSize(1.6); g_sweet_ep.SetMarkerColor(ROOT.kOrange+7)

frame_roc = ROOT.TH1D("frame_roc", f"ROC Curve & Efficiency vs. Purity (Sweet Spot Cut = {best_cut:.2f});False Positive Rate (or Signal Efficiency);True Positive Rate (or Purity)", 10, 0.0, 1.0)
frame_roc.SetMinimum(0.0); frame_roc.SetMaximum(1.05); frame_roc.Draw()
g_roc.Draw("L SAME"); g_eff_pur.Draw("L SAME")
g_sweet_roc.Draw("P SAME"); g_sweet_ep.Draw("P SAME")

leg8 = ROOT.TLegend(0.22, 0.18, 0.88, 0.42)
leg8.AddEntry(g_roc, "ROC: True Pos Rate vs False Pos Rate", "l")
leg8.AddEntry(g_eff_pur, "Purity vs Signal Efficiency Curve", "l")
leg8.AddEntry(g_sweet_roc, f"Sweet Spot (cvn#geq{best_cut:.2f}: Eff={best_tpr:.2f}, FPR={best_fpr:.2f}, Pur={best_pur:.2f})", "p")
leg8.Draw()
c.Print(pdf_path)

# ==============================================================================
# PAGE 5: Tightened Vertex X,Y,Z & Normalized Direction Cosines (p_i / |p|)
# ==============================================================================
c.Clear(); c.Divide(3, 2)
for idx, (ha, hf) in enumerate([(h_vtx_x_all, h_vtx_x_fid), (h_vtx_y_all, h_vtx_y_fid), (h_vtx_z_all, h_vtx_z_fid)]):
    c.cd(idx + 1); ROOT.gPad.SetGrid(); ROOT.gPad.SetLogy(0)
    style_h(ha, ROOT.kBlue+2); style_h(hf, ROOT.kTeal-1)
    ha.SetMaximum(ha.GetMaximum() * 1.18)
    ha.Draw("HIST"); hf.Draw("HIST SAME")
    if idx == 0:
        lv = ROOT.TLegend(0.45, 0.72, 0.88, 0.88)
        lv.AddEntry(ha, "All Reco", "l")
        lv.AddEntry(hf, "Fiducial Cut", "l")
        lv.Draw()

for idx, (ha, hm) in enumerate([(h_udir_x_all, h_udir_x_numu), (h_udir_y_all, h_udir_y_numu), (h_udir_z_all, h_udir_z_numu)]):
    c.cd(idx + 4); ROOT.gPad.SetGrid(); ROOT.gPad.SetLogy(0)
    style_h(ha, ROOT.kBlue+2); style_h(hm, ROOT.kRed+1)
    ha.SetMinimum(0.0); ha.SetMaximum(ha.GetMaximum() * 1.18)
    ha.Draw("HIST"); hm.Draw("HIST SAME")
    if idx == 0:
        ld = ROOT.TLegend(0.45, 0.72, 0.88, 0.88)
        ld.AddEntry(ha, "All Reco", "l")
        ld.AddEntry(hm, "True #nu_{#mu} CC", "l")
        ld.Draw()

c.Print(pdf_path)
c.Print(pdf_path + "]")
print("SUCCESS! Saved updated 5-page PDF to:", pdf_path, flush=True)