class Variable(object):
    def __init__(self, name_, caption_, nbins_, xmin_, xmax_):
        self.name = name_
        self.caption = caption_
        self.nbins = nbins_
        self.xmin = xmin_
        self.xmax = xmax_
    
variables = []

variables.append(Variable("dimuon_mass", r"$m_{\mu\mu}$ [GeV]", 100, 110, 150))
variables.append(Variable("dimuon_mass_res", r"$\Delta M_{\mu\mu}$ [GeV]", 100, 0, 4))
variables.append(Variable("dimuon_mass_res_rel", r"$\Delta M_{\mu\mu} / M_{\mu\mu}$ [GeV]", 100, 0, 0.04))
variables.append(Variable("dimuon_pt", r"$p_{T}(\mu\mu)$ [GeV]", 100, 0, 200))
variables.append(Variable("dimuon_eta", r"$\eta (\mu\mu)$", 100, -5, 5))
variables.append(Variable("dimuon_phi", r"$\phi (\mu\mu)$", 100, -3.2, 3.2))
variables.append(Variable("dimuon_dEta", r"$\Delta\eta (\mu\mu)$", 100, 0, 3.5))
variables.append(Variable("dimuon_dPhi", r"$\Delta\phi (\mu\mu)$", 100, 0, 4))
variables.append(Variable("dimuon_dR", r"$\Delta R (\mu\mu)$", 100, 0, 4))
variables.append(Variable("dimuon_cosThetaCS", r"$\cos\theta_{CS}$", 100, 0, 1))

variables.append(Variable("mu1_pt", r"$p_{T}(\mu_{1})$ [GeV]", 40, 20, 200))
variables.append(Variable("mu1_pt_over_mass", r"$p_{T}(\mu_{1})/M_{\mu\mu}$ [GeV]", 100, 0, 2))
variables.append(Variable("mu1_eta", r"$\eta (\mu_{1})$", 100, -2.5, 2.5))
variables.append(Variable("mu1_phi", r"$\phi (\mu_{1})$", 100, -3.2, 3.2))
variables.append(Variable("mu1_iso", r"iso$(\mu1)$", 50, 0, 0.3))

variables.append(Variable("mu2_pt", r"$p_{T}(\mu_{2})$ [GeV]", 40, 20, 120))
variables.append(Variable("mu2_pt_over_mass", r"$p_{T}(\mu_{2})/M_{\mu\mu}$ [GeV]", 100, 0, 2))
variables.append(Variable("mu2_eta", r"$\eta (\mu_{2})$", 100, -2.5, 2.5))
variables.append(Variable("mu2_phi", r"$\phi (\mu_{2})$", 100, -3.2, 3.2))
variables.append(Variable("mu2_iso", r"iso$(\mu2)$", 50, 0, 0.3))

variables.append(Variable("jet1_pt", r"$p_{T}(jet1)$ [GeV]", 100, 0, 200))
variables.append(Variable("jet1_eta", r"$\eta (jet1)$", 100, -4.7, 4.7))
variables.append(Variable("jet1_phi", r"$\phi (jet1)$", 100, -3.2, 3.2))
variables.append(Variable("jet1_qgl", r"$QGL (jet1)$", 10, 0, 1))
variables.append(Variable("deta_mumuj1", r"$\Delta\eta (\mu\mu, jet1)$", 100, 0, 3.5))
variables.append(Variable("dphi_mumuj1", r"$\Delta\phi (\mu\mu, jet1)$", 100, 0, 4))

variables.append(Variable("jet2_pt", r"$p_{T}(jet2)$ [GeV]", 100, 0, 150))
variables.append(Variable("jet2_eta", r"$\eta (jet2)$", 100, -4.7, 4.7))
variables.append(Variable("jet2_phi", r"$\phi (jet2)$", 100, -3.2, 3.2))
variables.append(Variable("jet2_qgl", r"$QGL (jet2)$", 10, 0, 1))
variables.append(Variable("deta_mumuj2", r"$\Delta\eta (\mu\mu, jet2)$", 100, 0, 3.5))
variables.append(Variable("dphi_mumuj2", r"$\Delta\phi (\mu\mu, jet2)$", 100, 0, 4))

variables.append(Variable("min_deta_mumuj", r"$min. \Delta\eta (\mu\mu, j)$", 100, 0, 8))
variables.append(Variable("min_dphi_mumuj", r"$min. \Delta\phi (\mu\mu, j)$", 100, 0, 3.3))

variables.append(Variable("jj_mass", r"$M(jj)$ [GeV]", 100, 0, 600))
variables.append(Variable("jj_pt", r"$p_{T}(jj)$ [GeV]", 100, 0, 150))
variables.append(Variable("jj_eta", r"$\eta (jj)$", 100, -4.7, 4.7))
variables.append(Variable("jj_phi", r"$\phi (jj)$", 100, -3.2, 3.2))
variables.append(Variable("jj_dEta", r"$\Delta\eta (jj)$", 100, 0, 3.5))
variables.append(Variable("jj_dPhi", r"$\Delta\phi (jj)$", 100, 0, 3.5))

variables.append(Variable("mmjj_mass", r"$M(\mu\mu jj)$ [GeV]", 100, 0, 1200))
variables.append(Variable("mmjj_pt", r"$p_{T}(\mu\mu jj)$ [GeV]", 100, 0, 150))
variables.append(Variable("mmjj_eta", r"$\eta (\mu\mu jj)$", 100, -7, 7))
variables.append(Variable("mmjj_phi", r"$\phi (\mu\mu jj)$", 100, -3.2, 3.2))

variables.append(Variable("zeppenfeld", r"zeppenfeld", 100, -5, 5))
variables.append(Variable("rpt", r"$R_{p_T}$", 100, 0, 1))

variables.append(Variable("njets", "njets", 10, 0, 10))
variables.append(Variable("nsoftjets2", "nsoftjets2", 20, 0, 20))
variables.append(Variable("nsoftjets5", "nsoftjets5", 20, 0, 20))

variables.append(Variable("htsoft2", "htsoft2", 30, 0, 60))
variables.append(Variable("htsoft5", "htsoft5", 30, 0, 60))

variables.append(Variable("npv", "npv", 50, 0, 50))
variables.append(Variable("met", r"$E_{T}^{miss.}$ [GeV]", 100, 0, 200))

variables.append(Variable("dnn_score", r"DNN score", 50, 0, 1))

variables.append(Variable("event", "event", 1, 0, 1))
variables.append(Variable("event_weight", "event_weight", 1, 0, 1))


#vars_unbin = ['event', 'event_weight',\
#              'dimuon_mass', 'dimuon_pt', 'dimuon_eta', 'dimuon_phi', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR',\
#              'mu1_pt', 'mu2_pt']

vars_unbin = [v.name for v in variables]