class Variable(object):
    def __init__(self, name_, caption_, nbins_, xmin_, xmax_):
        self.name = name_
        self.caption = caption_
        self.nbins = nbins_
        self.xmin = xmin_
        self.xmax = xmax_
    
variables = []

variables.append(Variable("dimuon_mass", r"$m_{\mu\mu}$ [GeV]", 50, 110, 150))
variables.append(Variable("dimuon_mass_res", r"$\Delta M_{\mu\mu}$ [GeV]", 50, 0, 4))
variables.append(Variable("dimuon_mass_res_rel", r"$\Delta M_{\mu\mu} / M_{\mu\mu}$ [GeV]", 50, 0, 0.04))
variables.append(Variable("dimuon_pt", r"$p_{T}(\mu\mu)$ [GeV]", 50, 0, 200))
variables.append(Variable("dimuon_eta", r"$\eta (\mu\mu)$", 50, -5, 5))
variables.append(Variable("dimuon_phi", r"$\phi (\mu\mu)$", 50, -3.2, 3.2))
variables.append(Variable("dimuon_dEta", r"$\Delta\eta (\mu\mu)$", 50, 0, 10))
variables.append(Variable("dimuon_dPhi", r"$\Delta\phi (\mu\mu)$", 50, 0, 4))
variables.append(Variable("dimuon_dR", r"$\Delta R (\mu\mu)$", 50, 0, 4))
variables.append(Variable("dimuon_cosThetaCS", r"$\cos\theta_{CS}$", 50, 0, 1))

variables.append(Variable("mu1_pt", r"$p_{T}(\mu_{1})$ [GeV]", 40, 20, 200))
variables.append(Variable("mu1_pt_over_mass", r"$p_{T}(\mu_{1})/M_{\mu\mu}$ [GeV]", 50, 0, 2))
variables.append(Variable("mu1_eta", r"$\eta (\mu_{1})$", 50, -2.5, 2.5))
variables.append(Variable("mu1_phi", r"$\phi (\mu_{1})$", 50, -3.2, 3.2))
variables.append(Variable("mu1_iso", r"iso$(\mu1)$", 50, 0, 0.3))

variables.append(Variable("mu2_pt", r"$p_{T}(\mu_{2})$ [GeV]", 40, 20, 120))
variables.append(Variable("mu2_pt_over_mass", r"$p_{T}(\mu_{2})/M_{\mu\mu}$ [GeV]", 50, 0, 2))
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

variables.append(Variable("mmj_min_dEta", r"$min. \Delta\eta (\mu\mu, j)$", 50, 0, 10))
variables.append(Variable("mmj_min_dPhi", r"$min. \Delta\phi (\mu\mu, j)$", 50, 0, 3.3))

variables.append(Variable("jj_mass", r"$M(jj)$ [GeV]", 50, 0, 600))
variables.append(Variable("jj_pt", r"$p_{T}(jj)$ [GeV]", 50, 0, 150))
variables.append(Variable("jj_eta", r"$\eta (jj)$", 50, -4.7, 4.7))
variables.append(Variable("jj_phi", r"$\phi (jj)$", 50, -3.2, 3.2))
variables.append(Variable("jj_dEta", r"$\Delta\eta (jj)$", 50, 0, 10))
variables.append(Variable("jj_dPhi", r"$\Delta\phi (jj)$", 50, 0, 3.5))

variables.append(Variable("mmjj_mass", r"$M(\mu\mu jj)$ [GeV]", 50, 0, 1200))
variables.append(Variable("mmjj_pt", r"$p_{T}(\mu\mu jj)$ [GeV]", 50, 0, 150))
variables.append(Variable("mmjj_eta", r"$\eta (\mu\mu jj)$", 50, -7, 7))
variables.append(Variable("mmjj_phi", r"$\phi (\mu\mu jj)$", 50, -3.2, 3.2))

variables.append(Variable("zeppenfeld", r"zeppenfeld", 50, -5, 5))
variables.append(Variable("rpt", r"$R_{p_T}$", 50, 0, 1))

variables.append(Variable("njets", "njets", 10, 0, 10))
variables.append(Variable("nsoftjets2", "nsoftjets2", 20, 0, 20))
variables.append(Variable("nsoftjets5", "nsoftjets5", 20, 0, 20))

variables.append(Variable("htsoft2", "htsoft2", 30, 0, 60))
variables.append(Variable("htsoft5", "htsoft5", 30, 0, 60))

variables.append(Variable("npv", "npv", 50, 0, 50))
variables.append(Variable("met", r"$E_{T}^{miss.}$ [GeV]", 50, 0, 200))

variables.append(Variable("dnn_score", r"DNN score", 50, 0, 1))
variables.append(Variable("btag_wgt", r"b-tag weight", 50, 0, 2))

variables.append(Variable("event", "event", 1, 0, 1))
variables.append(Variable("run", "run", 1, 0, 1))


all_columns = ['mu1_pt', 'mmj_min_dPhi', 'mu2_iso', 'mu1_pt_over_mass', 'dimuon_mass_res_rel', 'dimuon_phi', 'jet1_qgl', 'mu2_pt_over_mass', 'wgt_qgl_wgt_off', 'dimuon_mass', 'jet2_qgl', 'rpt', 'wgt_genwgt_down', 'jet1_phi', 'jj_mass', 'jet1_eta', 'wgt_muSF_down', 'jj_dEta', 'wgt_btag_wgt_down', 'wgt_l1prefiring_wgt_off', 'wgt_nnlops_off', 'jet2_eta', 'wgt_pu_wgt_down', 'mu1_phi', 'mu2_pt', 'jet1_puid', 'wgt_l1prefiring_wgt_up', 'jj_phi', 'mmj2_dPhi', 'wgt_lumi_down', 'wgt_puid_wgt_off', 'mmj_min_dEta', 'dimuon_dEta', 'wgt_nnlops_up', 'wgt_pu_wgt_off', 'jj_pt', 'jet1_id', 'wgt_nominal', 'mu2_eta', 'wgt_muSF_up', 'wgt_muSF_off', 'wgt_l1prefiring_wgt_down', 'mmj1_dPhi', 'dimuon_cosThetaCS', 'wgt_genwgt_up', 'btag_wgt', 'nsoftjets2', 'dimuon_dPhi', 'jet2_pt', 'dimuon_eta', 'mu1_eta', 'jet2_puid', 'event', 'jet2_id', 'wgt_btag_wgt_up', 'njets', 'htsoft2', 'dimuon_dR', 'mmjj_pt', 'npv', 'wgt_qgl_wgt_up', 'mu1_iso', 'wgt_btag_wgt_off', 'wgt_pu_wgt_up', 'wgt_nnlops_down', 'jet2_phi', 'wgt_genwgt_off', 'mmj2_dEta', 'wgt_lumi_up', 'mmjj_eta', 'wgt_puid_wgt_up', 'jet1_pt', 'nsoftjets5', 'mu2_phi', 'met', 'dnn_score', 'mmj1_dEta', 'dimuon_mass_res', 'mmjj_mass', 'mmjj_phi', 'htsoft5', 'dimuon_pt', 'jj_eta', 'jj_dPhi', 'wgt_lumi_off', 'wgt_puid_wgt_down', 'zeppenfeld', 'wgt_qgl_wgt_down']