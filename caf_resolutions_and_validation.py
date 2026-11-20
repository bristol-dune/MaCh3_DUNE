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
print("Opening CAF file:", caf_path)
f = ROOT.TFile.Open(caf_path, "READ")
caf_tree = f.Get("cafTree")

max_events = 250000 if len(sys.argv) <= 1 else (-1 if sys.argv[1].lower() == "all" else int(sys.argv[1]))

# Energy bins for Resolution vs. True Energy and Selection Validation
e_bins = [0.2, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0, 7.5, 10.0]
n_ebins = len(e_bins) - 1
e_bins_arr = array('d', e_bins)

# 1. Book 1D Residual Histograms (All Reco vs True nu_mu CC)
h_res_lep_all  = ROOT.TH1D("h_res_lep_all", "Energy Residuals: All Reconstructed Events;(E_{reco} - E_{true}) / E_{true};Events", 80, -1.0, 1.0)
h_res_ecal_all = ROOT.TH1D("h_res_ecal_all", "Energy Residuals: All Reconstructed Events;(E_{reco} - E_{true}) / E_{true};Events", 80, -1.0, 1.0)

h_res_lep_numu  = ROOT.TH1D("h_res_lep_numu", "Energy Residuals: True #nu_{#mu} CC Events (|PDG|=14, CC);(E_{reco} - E_{true}) / E_{true};Events", 80, -1.0, 1.0)
h_res_ecal_numu = ROOT.TH1D("h_res_ecal_numu", "Energy Residuals: True #nu_{#mu} CC Events (|PDG|=14, CC);(E_{reco} - E_{true}) / E_{true};Events", 80, -1.0, 1.0)

# Muon-specific energy estimators on True nu_mu CC
h_res_rng_numu = ROOT.TH1D("h_res_rng_numu", "Muon Energy Versions on True #nu_{#mu} CC;(E_{reco} - E_{true}) / E_{true};Events", 80, -1.0, 1.0)
h_res_mcs_numu = ROOT.TH1D("h_res_mcs_numu", "Muon Energy Versions on True #nu_{#mu} CC;(E_{reco} - E_{true}) / E_{true};Events", 80, -1.0, 1.0)

# Direction (cos theta_z) residuals on True nu_mu CC
h_dir_trk_numu = ROOT.TH1D("h_dir_trk_numu", "Direction (cos#theta_{z}) Residuals on True #nu_{#mu} CC;cos#theta_{z,reco} - cos#theta_{z,true};Events", 80, -1.0, 1.0)
h_dir_shw_numu = ROOT.TH1D("h_dir_shw_numu", "Direction (cos#theta_{z}) Residuals on True #nu_{#mu} CC;cos#theta_{z,reco} - cos#theta_{z,true};Events", 80, -1.0, 1.0)

# CVN score distributions by true interaction type
h_cvn_numu_sig = ROOT.TH1D("h_cvn_sig", "CVN #nu_{#mu} Classifier Score by True Type;cvn.numu Score;Events", 50, 0.0, 1.0)
h_cvn_numu_nue = ROOT.TH1D("h_cvn_nue", "CVN #nu_{#mu} Classifier Score by True Type;cvn.numu Score;Events", 50, 0.0, 1.0)
h_cvn_numu_nc  = ROOT.TH1D("h_cvn_nc",  "CVN #nu_{#mu} Classifier Score by True Type;cvn.numu Score;Events", 50, 0.0, 1.0)

# New kinematics distributions (Vertex & Momentum)
h_vtx_x_all = ROOT.TH1D("h_vtx_x_all", "True Vertex X;mc.nu.vtx.x [cm];Events", 70, -700, 700)
h_vtx_x_sel = ROOT.TH1D("h_vtx_x_sel", "True Vertex X;mc.nu.vtx.x [cm];Events", 70, -700, 700)
h_vtx_y_all = ROOT.TH1D("h_vtx_y_all", "True Vertex Y;mc.nu.vtx.y [cm];Events", 70, -700, 700)
h_vtx_y_sel = ROOT.TH1D("h_vtx_y_sel", "True Vertex Y;mc.nu.vtx.y [cm];Events", 70, -700, 700)
h_vtx_z_all = ROOT.TH1D("h_vtx_z_all", "True Vertex Z;mc.nu.vtx.z [cm];Events", 75, -1500, 1500)
h_vtx_z_sel = ROOT.TH1D("h_vtx_z_sel", "True Vertex Z;mc.nu.vtx.z [cm];Events", 75, -1500, 1500)

h_mom_x_all  = ROOT.TH1D("h_mom_x_all", "True Momentum p_{x};mc.nu.momentum.x [GeV/c];Events", 80, -5, 5)
h_mom_x_numu = ROOT.TH1D("h_mom_x_numu", "True Momentum p_{x};mc.nu.momentum.x [GeV/c];Events", 80, -5, 5)
h_mom_y_all  = ROOT.TH1D("h_mom_y_all", "True Momentum p_{y};mc.nu.momentum.y [GeV/c];Events", 80, -5, 5)
h_mom_y_numu = ROOT.TH1D("h_mom_y_numu", "True Momentum p_{y};mc.nu.momentum.y [GeV/c];Events", 80, -5, 5)
h_mom_z_all  = ROOT.TH1D("h_mom_z_all", "True Momentum p_{z};mc.nu.momentum.z [GeV/c];Events", 80, -5, 5)
h_mom_z_numu = ROOT.TH1D("h_mom_z_numu", "True Momentum p_{z};mc.nu.momentum.z [GeV/c];Events", 80, -5, 5)

# Lists to store residuals per true-energy bin for exact 68% quantile resolution calculation
res_bins_lep  = [[] for _ in range(n_ebins)]
res_bins_ecal = [[] for _ in range(n_ebins)]
res_bins_rng  = [[] for _ in range(n_ebins)]
res_bins_mcs  = [[] for _ in range(n_ebins)]

# Counters for Selection Efficiency and Purity per energy bin
n_true_numu_bin     = [0] * n_ebins
n_sel_all_bin       = [0] * n_ebins
n_sel_true_numu_bin = [0] * n_ebins

def find_ebin(e_val):
    for i in range(n_ebins):
        if e_bins[i] <= e_val < e_bins[i+1]:
            return i
    return -1

# Reusable event filter function matching your supervisor's requirements
def pass_filter(is_reco_ok, pdg, is_cc, true_e, reco_e, cvn_numu,
                req_abs_pdg=None, req_cc=None, min_reco_e=None, min_cvn_numu=None):
    if not is_reco_ok:
        return False
    if req_abs_pdg is not None and abs(pdg) != req_abs_pdg:
        return False
    if req_cc is not None and bool(is_cc) != req_cc:
        return False
    if min_reco_e is not None and reco_e < min_reco_e:
        return False
    if min_cvn_numu is not None and cvn_numu < min_cvn_numu:
        return False
    return True

print("Looping over cafTree events...")
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
    true_dir_z = (pz / p_mag) if p_mag > 0 else -999.0

    if entry.rec.common.ixn.pandora.size() != 1:
        continue

    r = entry.rec.common.ixn.pandora[0]
    e_lep  = float(r.Enu.lep_calo)
    e_ecal = float(r.Enu.e_calo)
    e_rng  = float(r.Enu.mu_range)
    e_mcs  = float(r.Enu.mu_mcs)

    dir_trk_z = float(r.dir.lngtrk.z)
    dir_shw_z = float(r.dir.heshw.z)

    cvn_nue  = float(r.nuhyp.cvn.nue)
    cvn_numu = float(r.nuhyp.cvn.numu)
    cvn_nc   = float(r.nuhyp.cvn.nc)

    # Fill all-reco kinematics & residuals
    h_vtx_x_all.Fill(vx); h_vtx_y_all.Fill(vy); h_vtx_z_all.Fill(vz)
    h_mom_x_all.Fill(px); h_mom_y_all.Fill(py); h_mom_z_all.Fill(pz)

    res_lep  = (e_lep - true_e) / true_e if (e_lep > 0 and true_e > 0) else None
    res_ecal = (e_ecal - true_e) / true_e if (e_ecal > 0 and true_e > 0) else None
    res_rng  = (e_rng - true_e) / true_e if (e_rng > 0 and true_e > 0) else None
    res_mcs  = (e_mcs - true_e) / true_e if (e_mcs > 0 and true_e > 0) else None

    if res_lep is not None:  h_res_lep_all.Fill(res_lep)
    if res_ecal is not None: h_res_ecal_all.Fill(res_ecal)

    # CVN distributions by true channel
    if not true_iscc:
        h_cvn_numu_nc.Fill(cvn_numu)
    elif abs(true_pdg) == 14:
        h_cvn_numu_sig.Fill(cvn_numu)
    elif abs(true_pdg) == 12:
        h_cvn_numu_nue.Fill(cvn_numu)

    # Selected nu_mu-like events (cvn.numu >= 0.5)
    is_sel_numu = pass_filter(True, true_pdg, true_iscc, true_e, e_lep, cvn_numu, min_cvn_numu=0.5)
    if is_sel_numu:
        h_vtx_x_sel.Fill(vx); h_vtx_y_sel.Fill(vy); h_vtx_z_sel.Fill(vz)

    # True nu_mu CC events (|pdg| == 14 and iscc == True)
    is_true_numu_cc = pass_filter(True, true_pdg, true_iscc, true_e, e_lep, cvn_numu, req_abs_pdg=14, req_cc=True)
    if is_true_numu_cc:
        h_mom_x_numu.Fill(px); h_mom_y_numu.Fill(py); h_mom_z_numu.Fill(pz)
        if res_lep is not None:  h_res_lep_numu.Fill(res_lep)
        if res_ecal is not None: h_res_ecal_numu.Fill(res_ecal)
        if res_rng is not None:  h_res_rng_numu.Fill(res_rng)
        if res_mcs is not None:  h_res_mcs_numu.Fill(res_mcs)
        if dir_trk_z > -990 and true_dir_z > -990:
            h_dir_trk_numu.Fill(dir_trk_z - true_dir_z)
        if dir_shw_z > -990 and true_dir_z > -990:
            h_dir_shw_numu.Fill(dir_shw_z - true_dir_z)

    # Binning by True Energy for Resolution vs Energy & Selection Validation
    eb_idx = find_ebin(true_e)
    if eb_idx >= 0:
        if is_true_numu_cc:
            n_true_numu_bin[eb_idx] += 1
            if res_lep is not None and -1.0 <= res_lep <= 1.5:   res_bins_lep[eb_idx].append(res_lep)
            if res_ecal is not None and -1.0 <= res_ecal <= 1.5: res_bins_ecal[eb_idx].append(res_ecal)
            if res_rng is not None and -1.0 <= res_rng <= 1.5:   res_bins_rng[eb_idx].append(res_rng)
            if res_mcs is not None and -1.0 <= res_mcs <= 1.5:   res_bins_mcs[eb_idx].append(res_mcs)
        if is_sel_numu:
            n_sel_all_bin[eb_idx] += 1
            if is_true_numu_cc:
                n_sel_true_numu_bin[eb_idx] += 1

    if (n_proc + 1) % 50000 == 0:
        print(f"  Processed {n_proc + 1} events...")

# Helper to build TGraphErrors for Resolution (68% half-width) & Median Bias vs True Energy
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
        g.SetLineColor(color)
        g.SetMarkerColor(color)
        g.SetMarkerStyle(marker)
        g.SetLineWidth(2)
        g.SetLineStyle(style)
    return g_res, g_bias

g_res_lep,  g_bias_lep  = make_res_and_bias_graphs(res_bins_lep,  ROOT.kBlue+1,   20, 1)
g_res_ecal, g_bias_ecal = make_res_and_bias_graphs(res_bins_ecal, ROOT.kRed+1,    21, 1)
g_res_rng,  g_bias_rng  = make_res_and_bias_graphs(res_bins_rng,  ROOT.kGreen+2,  22, 2)
g_res_mcs,  g_bias_mcs  = make_res_and_bias_graphs(res_bins_mcs,  ROOT.kOrange+7, 23, 2)

# Helper to style 1D histograms
def style_h(h, color, ls=1):
    h.SetLineColor(color)
    h.SetLineWidth(2)
    h.SetLineStyle(ls)

os.makedirs(os.path.expanduser("~/todays_pdfs"), exist_ok=True)
pdf_path = os.path.expanduser("~/todays_pdfs/CAF_Resolutions_and_Validation.pdf")

c = ROOT.TCanvas("c", "CAF Analysis", 1200, 550)
c.Print(pdf_path + "[")

# PAGE 1: Energy Residuals (lep_calo vs e_calo) on All Reco and True nu_mu CC
c.Clear(); c.Divide(2, 1)
c.cd(1); ROOT.gPad.SetGrid()
style_h(h_res_lep_all, ROOT.kBlue+1); style_h(h_res_ecal_all, ROOT.kRed+1)
ymax = max(h_res_lep_all.GetMaximum(), h_res_ecal_all.GetMaximum()) * 1.15
h_res_lep_all.SetMaximum(ymax)
h_res_lep_all.Draw("HIST"); h_res_ecal_all.Draw("HIST SAME")
leg1 = ROOT.TLegend(0.52, 0.72, 0.88, 0.88)
leg1.AddEntry(h_res_lep_all, "lep_calo (Muon hyp)", "l")
leg1.AddEntry(h_res_ecal_all, "e_calo (Electron hyp)", "l")
leg1.Draw()

c.cd(2); ROOT.gPad.SetGrid()
style_h(h_res_lep_numu, ROOT.kBlue+1); style_h(h_res_ecal_numu, ROOT.kRed+1)
ymax = max(h_res_lep_numu.GetMaximum(), h_res_ecal_numu.GetMaximum()) * 1.15
h_res_lep_numu.SetMaximum(ymax)
h_res_lep_numu.Draw("HIST"); h_res_ecal_numu.Draw("HIST SAME")
leg2 = ROOT.TLegend(0.52, 0.72, 0.88, 0.88)
leg2.AddEntry(h_res_lep_numu, "lep_calo on True #nu_{#mu} CC", "l")
leg2.AddEntry(h_res_ecal_numu, "e_calo on True #nu_{#mu} CC", "l")
leg2.Draw()
c.Print(pdf_path)

# PAGE 2: Muon-specific Energy Versions (mu_range vs mu_mcs) & Direction Residuals
c.Clear(); c.Divide(2, 1)
c.cd(1); ROOT.gPad.SetGrid()
style_h(h_res_rng_numu, ROOT.kGreen+2); style_h(h_res_mcs_numu, ROOT.kOrange+7)
ymax = max(h_res_rng_numu.GetMaximum(), h_res_mcs_numu.GetMaximum(), h_res_lep_numu.GetMaximum()) * 1.15
h_res_rng_numu.SetMaximum(ymax)
h_res_rng_numu.Draw("HIST"); h_res_mcs_numu.Draw("HIST SAME"); h_res_lep_numu.Draw("HIST SAME")
leg3 = ROOT.TLegend(0.48, 0.68, 0.88, 0.88)
leg3.AddEntry(h_res_rng_numu, "Enu.mu_range (Contained)", "l")
leg3.AddEntry(h_res_mcs_numu, "Enu.mu_mcs (Scattering)", "l")
leg3.AddEntry(h_res_lep_numu, "Enu.lep_calo (Combined)", "l")
leg3.Draw()

c.cd(2); ROOT.gPad.SetGrid()
style_h(h_dir_trk_numu, ROOT.kBlue+1); style_h(h_dir_shw_numu, ROOT.kRed+1)
ymax = max(h_dir_trk_numu.GetMaximum(), h_dir_shw_numu.GetMaximum()) * 1.15
h_dir_trk_numu.SetMaximum(ymax)
h_dir_trk_numu.Draw("HIST"); h_dir_shw_numu.Draw("HIST SAME")
leg4 = ROOT.TLegend(0.48, 0.72, 0.88, 0.88)
leg4.AddEntry(h_dir_trk_numu, "dir.lngtrk.z (Longest Track)", "l")
leg4.AddEntry(h_dir_shw_numu, "dir.heshw.z (Highest-E Shower)", "l")
leg4.Draw()
c.Print(pdf_path)

# PAGE 3: Energy Resolution & Scale Bias as a Function of True Energy (True nu_mu CC)
c.Clear(); c.Divide(2, 1)
c.cd(1); ROOT.gPad.SetGrid()
frame_res = ROOT.TH1D("frame_res", "Energy Resolution vs. True Energy (True #nu_{#mu} CC);True Neutrino Energy E_{true} [GeV];Energy Resolution #sigma_{E} / E_{true} (68% Width)", 10, 0.0, 10.0)
frame_res.SetMinimum(0.0); frame_res.SetMaximum(0.65); frame_res.Draw()
g_res_lep.Draw("LP SAME"); g_res_ecal.Draw("LP SAME"); g_res_rng.Draw("LP SAME"); g_res_mcs.Draw("LP SAME")
leg5 = ROOT.TLegend(0.50, 0.66, 0.88, 0.88)
leg5.AddEntry(g_res_lep,  "lep_calo (68% width)", "lp")
leg5.AddEntry(g_res_ecal, "e_calo (68% width)", "lp")
leg5.AddEntry(g_res_rng,  "mu_range (68% width)", "lp")
leg5.AddEntry(g_res_mcs,  "mu_mcs (68% width)", "lp")
leg5.Draw()

c.cd(2); ROOT.gPad.SetGrid()
frame_bias = ROOT.TH1D("frame_bias", "Energy Scale Bias vs. True Energy (True #nu_{#mu} CC);True Neutrino Energy E_{true} [GeV];Median (E_{reco} - E_{true}) / E_{true}", 10, 0.0, 10.0)
frame_bias.SetMinimum(-0.6); frame_bias.SetMaximum(0.4); frame_bias.Draw()
g_bias_lep.Draw("LP SAME"); g_bias_ecal.Draw("LP SAME"); g_bias_rng.Draw("LP SAME"); g_bias_mcs.Draw("LP SAME")
leg6 = ROOT.TLegend(0.50, 0.66, 0.88, 0.88)
leg6.AddEntry(g_bias_lep,  "lep_calo (Median Bias)", "lp")
leg6.AddEntry(g_bias_ecal, "e_calo (Median Bias)", "lp")
leg6.AddEntry(g_bias_rng,  "mu_range (Median Bias)", "lp")
leg6.AddEntry(g_bias_mcs,  "mu_mcs (Median Bias)", "lp")
leg6.Draw()
c.Print(pdf_path)

# PAGE 4: Selection Validation (CVN Score & Efficiency/Purity vs True Energy)
c.Clear(); c.Divide(2, 1)
c.cd(1); ROOT.gPad.SetGrid(); ROOT.gPad.SetLogy(1)
style_h(h_cvn_numu_sig, ROOT.kBlue+1); style_h(h_cvn_numu_nue, ROOT.kRed+1); style_h(h_cvn_numu_nc, ROOT.kGreen+2)
h_cvn_numu_sig.SetMinimum(0.8)
h_cvn_numu_sig.Draw("HIST"); h_cvn_numu_nue.Draw("HIST SAME"); h_cvn_numu_nc.Draw("HIST SAME")
leg7 = ROOT.TLegend(0.42, 0.68, 0.88, 0.88)
leg7.AddEntry(h_cvn_numu_sig, "True #nu_{#mu} CC (Signal)", "l")
leg7.AddEntry(h_cvn_numu_nue, "True #nu_{e} CC (Background)", "l")
leg7.AddEntry(h_cvn_numu_nc,  "True NC (Background)", "l")
leg7.Draw()

c.cd(2); ROOT.gPad.SetGrid(); ROOT.gPad.SetLogy(0)
g_eff = ROOT.TGraphErrors(); g_pur = ROOT.TGraphErrors()
pt = 0
for i in range(n_ebins):
    xc = 0.5 * (e_bins[i] + e_bins[i+1])
    xe = 0.5 * (e_bins[i+1] - e_bins[i])
    eff = n_sel_true_numu_bin[i] / float(n_true_numu_bin[i]) if n_true_numu_bin[i] > 0 else 0.0
    pur = n_sel_true_numu_bin[i] / float(n_sel_all_bin[i]) if n_sel_all_bin[i] > 0 else 0.0
    g_eff.SetPoint(pt, xc, eff); g_eff.SetPointError(pt, xe, 0.0)
    g_pur.SetPoint(pt, xc, pur); g_pur.SetPointError(pt, xe, 0.0)
    pt += 1
for g, col, mrk in [(g_eff, ROOT.kBlue+1, 20), (g_pur, ROOT.kOrange+7, 21)]:
    g.SetLineColor(col); g.SetMarkerColor(col); g.SetMarkerStyle(mrk); g.SetLineWidth(2)

frame_val = ROOT.TH1D("frame_val", "Selection Validation vs. True Energy (cvn.numu #geq 0.5);True Neutrino Energy E_{true} [GeV];Efficiency / Purity", 10, 0.0, 10.0)
frame_val.SetMinimum(0.0); frame_val.SetMaximum(1.05); frame_val.Draw()
g_eff.Draw("LP SAME"); g_pur.Draw("LP SAME")
leg8 = ROOT.TLegend(0.50, 0.20, 0.88, 0.38)
leg8.AddEntry(g_eff, "Selection Efficiency (#nu_{#mu} CC)", "lp")
leg8.AddEntry(g_pur, "Selection Purity (#nu_{#mu} CC)", "lp")
leg8.Draw()
c.Print(pdf_path)

# PAGE 5: Vertex & Momentum Distributions
c.Clear(); c.Divide(3, 2)
for idx, (ha, hs) in enumerate([(h_vtx_x_all, h_vtx_x_sel), (h_vtx_y_all, h_vtx_y_sel), (h_vtx_z_all, h_vtx_z_sel)]):
    c.cd(idx + 1); ROOT.gPad.SetGrid(); ROOT.gPad.SetLogy(0)
    style_h(ha, ROOT.kBlue+2); style_h(hs, ROOT.kTeal-1)
    ha.Draw("HIST"); hs.Draw("HIST SAME")
for idx, (ha, hm) in enumerate([(h_mom_x_all, h_mom_x_numu), (h_mom_y_all, h_mom_y_numu), (h_mom_z_all, h_mom_z_numu)]):
    c.cd(idx + 4); ROOT.gPad.SetGrid(); ROOT.gPad.SetLogy(1)
    style_h(ha, ROOT.kBlue+2); style_h(hm, ROOT.kRed+1)
    ha.SetMinimum(0.8); ha.Draw("HIST"); hm.Draw("HIST SAME")
c.Print(pdf_path)

c.Print(pdf_path + "]")
print("SUCCESS! Saved 5-page PDF to:", pdf_path)