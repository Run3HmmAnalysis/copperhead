class Variable(object):
    def __init__(self, name_, caption_, nbins_, xmin_, xmax_):
        self.name = name_
        self.caption = caption_
        self.nbins = nbins_
        self.xmin = xmin_
        self.xmax = xmax_


variables = []
variables.append(Variable("dimuon_mass", r"$m_{\mu\mu}$ [GeV]", 50, 110, 150))
variables.append(Variable("dimuon_mass_res", r"$\Delta M_{\mu\mu}$ [GeV]", 50, 0, 10))
variables.append(
    Variable(
        "dimuon_mass_res_rel", r"$\Delta M_{\mu\mu} / M_{\mu\mu}$ [GeV]", 50, 0, 0.1
    )
)
variables.append(
    Variable("dimuon_ebe_mass_res", r"$\Delta M_{\mu\mu}$ [GeV]", 50, 0, 4)
)
variables.append(
    Variable(
        "dimuon_ebe_mass_res_rel",
        r"$\Delta M_{\mu\mu} / M_{\mu\mu}$ [GeV]",
        50,
        0,
        0.04,
    )
)
variables.append(Variable("dimuon_pt", r"$p_{T}(\mu\mu)$ [GeV]", 50, 0, 200))
variables.append(Variable("dimuon_pt_log", r"$log(p_{T}(\mu\mu))$", 50, 0, 200))
variables.append(Variable("dimuon_eta", r"$\eta (\mu\mu)$", 50, -5, 5))
variables.append(Variable("dimuon_phi", r"$\phi (\mu\mu)$", 50, -3.2, 3.2))
variables.append(Variable("dimuon_dEta", r"$\Delta\eta (\mu\mu)$", 50, 0, 10))
variables.append(Variable("dimuon_dPhi", r"$\Delta\phi (\mu\mu)$", 50, 0, 4))
variables.append(Variable("dimuon_dR", r"$\Delta R (\mu\mu)$", 50, 0, 4))
variables.append(Variable("dimuon_cos_theta_cs", r"$\cos\theta_{CS}$", 50, 0, 1))
variables.append(Variable("dimuon_phi_cs", r"$\phi_{CS}$", 50, 0, 6.5))
variables.append(Variable("mu1_pt", r"$p_{T}(\mu_{1})$ [GeV]", 40, 20, 200))
variables.append(
    Variable("mu1_pt_over_mass", r"$p_{T}(\mu_{1})/M_{\mu\mu}$ [GeV]", 50, 0, 2)
)
variables.append(Variable("mu1_eta", r"$\eta (\mu_{1})$", 50, -2.5, 2.5))
variables.append(Variable("mu1_phi", r"$\phi (\mu_{1})$", 50, -3.2, 3.2))
variables.append(Variable("mu1_iso", r"iso$(\mu1)$", 50, 0, 0.3))
variables.append(Variable("mu2_pt", r"$p_{T}(\mu_{2})$ [GeV]", 40, 20, 120))
variables.append(
    Variable("mu2_pt_over_mass", r"$p_{T}(\mu_{2})/M_{\mu\mu}$ [GeV]", 50, 0, 2)
)
variables.append(Variable("mu2_eta", r"$\eta (\mu_{2})$", 50, -2.5, 2.5))
variables.append(Variable("mu2_phi", r"$\phi (\mu_{2})$", 50, -3.2, 3.2))
variables.append(Variable("mu2_iso", r"iso$(\mu2)$", 50, 0, 0.3))
variables.append(Variable("jet1_pt", r"$p_{T}(jet1)$ [GeV]", 50, 0, 200))
variables.append(Variable("jet1_eta", r"$\eta (jet1)$", 50, -4.7, 4.7))
variables.append(Variable("jet1_phi", r"$\phi (jet1)$", 50, -3.2, 3.2))
variables.append(Variable("jet1_qgl", r"$QGL (jet1)$", 10, 0, 1))
variables.append(Variable("jet1_id", "jet1 ID", 8, 0, 8))
variables.append(Variable("jet1_puid", "jet1 PUID", 8, 0, 8))
variables.append(Variable("mmj1_dEta", r"$\Delta\eta (\mu\mu, jet1)$", 50, 0, 10))
variables.append(Variable("mmj1_dPhi", r"$\Delta\phi (\mu\mu, jet1)$", 50, 0, 4))
variables.append(Variable("jet2_pt", r"$p_{T}(jet2)$ [GeV]", 50, 0, 150))
variables.append(Variable("jet2_eta", r"$\eta (jet2)$", 50, -4.7, 4.7))
variables.append(Variable("jet2_phi", r"$\phi (jet2)$", 50, -3.2, 3.2))
variables.append(Variable("jet2_qgl", r"$QGL (jet2)$", 10, 0, 1))
variables.append(Variable("jet2_id", "jet2 ID", 8, 0, 8))
variables.append(Variable("jet2_puid", "jet2 PUID", 8, 0, 8))
variables.append(Variable("mmj2_dEta", r"$\Delta\eta (\mu\mu, jet2)$", 50, 0, 10))
variables.append(Variable("mmj2_dPhi", r"$\Delta\phi (\mu\mu, jet2)$", 50, 0, 4))
variables.append(Variable("jet1_has_matched_muon", "jet1_has_matched_muon", 4, -1, 3))
variables.append(Variable("jet2_has_matched_muon", "jet2_has_matched_muon", 4, -1, 3))
variables.append(Variable("jet1_matched_muon_dr", "jet1_matched_muon_dr", 50, 0, 0.8))
variables.append(Variable("jet2_matched_muon_dr", "jet2_matched_muon_dr", 50, 0, 0.8))
variables.append(Variable("jet1_matched_muon_pt", "jet1_matched_muon_pt", 50, 0, 200))
variables.append(Variable("jet2_matched_muon_pt", "jet2_matched_muon_pt", 50, 0, 200))
variables.append(Variable("jet1_matched_muon_iso", "jet1_matched_muon_iso", 100, 0, 5))
variables.append(Variable("jet2_matched_muon_iso", "jet2_matched_muon_iso", 100, 0, 5))
variables.append(Variable("jet1_matched_muon_id", "jet1_matched_muon_id", 4, -1, 3))
variables.append(Variable("jet2_matched_muon_id", "jet2_matched_muon_id", 4, -1, 3))
variables.append(Variable("mmj_min_dEta", r"$min. \Delta\eta (\mu\mu, j)$", 50, 0, 10))
variables.append(Variable("mmj_min_dPhi", r"$min. \Delta\phi (\mu\mu, j)$", 50, 0, 3.3))
variables.append(Variable("jj_mass", r"$M(jj)$ [GeV]", 50, 0, 600))
variables.append(Variable("jj_mass_log", r"$log M(jj)$", 50, 0, 600))
variables.append(Variable("jj_pt", r"$p_{T}(jj)$ [GeV]", 50, 0, 150))
variables.append(Variable("jj_eta", r"$\eta (jj)$", 50, -4.7, 4.7))
variables.append(Variable("jj_phi", r"$\phi (jj)$", 50, -3.2, 3.2))
variables.append(Variable("jj_dEta", r"$\Delta\eta (jj)$", 50, 0, 10))
variables.append(Variable("jj_dPhi", r"$\Delta\phi (jj)$", 50, 0, 3.5))
variables.append(Variable("mmjj_mass", r"$M(\mu\mu jj)$ [GeV]", 50, 0, 1200))
variables.append(Variable("mmjj_pt", r"$p_{T}(\mu\mu jj)$ [GeV]", 50, 0, 150))
variables.append(Variable("mmjj_eta", r"$\eta (\mu\mu jj)$", 50, -7, 7))
variables.append(Variable("mmjj_phi", r"$\phi (\mu\mu jj)$", 50, -3.2, 3.2))
variables.append(Variable("ll_zstar_log", r"ll_zstar_log", 50, -10, 10))
variables.append(Variable("zeppenfeld", r"zeppenfeld", 50, -5, 5))
variables.append(Variable("rpt", r"$R_{p_T}$", 50, 0, 1))
variables.append(Variable("njets", "njets", 10, 0, 10))
variables.append(Variable("nsoftjets2", "nsoftjets2", 20, 0, 20))
variables.append(Variable("nsoftjets5", "nsoftjets5", 20, 0, 20))
variables.append(Variable("htsoft2", "htsoft2", 30, 0, 60))
variables.append(Variable("htsoft5", "htsoft5", 30, 0, 60))
variables.append(Variable("npv", "npv", 50, 0, 50))
variables.append(Variable("nTrueInt", "nTrueInt", 50, 0, 50))
variables.append(Variable("met", r"$E_{T}^{miss.}$ [GeV]", 50, 0, 200))
variables.append(Variable("btag_wgt", r"b-tag weight", 50, 0, 2))
variables.append(Variable("event", "event", 100, 0, 10000000))
variables.append(Variable("run", "run", 1, 0, 1))

variables_lookup = {}
for v in variables:
    variables_lookup[v.name] = v
