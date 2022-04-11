event_branches = ["run", "event", "luminosityBlock", "genWeight"]
muon_branches = [
    "nMuon",
    "Muon_pt",
    "Muon_ptErr",
    "Muon_eta",
    "Muon_phi",
    "Muon_mass",
    "Muon_charge",
    "Muon_pfRelIso04_all",
    # 'Muon_dxybs',
    "Muon_fsrPhotonIdx",
    "Muon_mediumId",
    "Muon_genPartIdx",
    "Muon_nTrackerLayers",
]
fsr_branches = [
    "nFsrPhoton",
    "FsrPhoton_pt",
    "FsrPhoton_eta",
    "FsrPhoton_phi",
    "FsrPhoton_relIso03",
    "FsrPhoton_dROverEt2",
]
jet_branches = [
    "nJet",
    "Jet_pt",
    "Jet_eta",
    "Jet_phi",
    "Jet_mass",
    "Jet_qgl",
    "Jet_jetId",
    "Jet_puId",
    "Jet_rawFactor",
    "Jet_hadronFlavour",
    "Jet_partonFlavour",
    "Jet_muonIdx1",
    "Jet_muonIdx2",
    "Jet_btagDeepB",
    "Jet_genJetIdx",
]
genjet_branches = ["nGenJet", "GenJet_pt", "GenJet_eta", "GenJet_phi", "GenJet_mass"]
sajet_branches = [
    "nSoftActivityJet",
    "SoftActivityJet_pt",
    "SoftActivityJet_eta",
    "SoftActivityJet_phi",
    "SoftActivityJetNjets2",
    "SoftActivityJetNjets5",
    "SoftActivityJetHT2",
    "SoftActivityJetHT5",
]
vtx_branches = ["Pileup_nTrueInt", "PV_npvsGood", "PV_npvs"]
genpart_branches = [
    "nGenPart",
    "GenPart_pt",
    "GenPart_eta",
    "GenPart_phi",
    "GenPart_pdgId",
]
trigobj_branches = [
    "nTrigObj",
    "TrigObj_pt",
    "TrigObj_l1pt",
    "TrigObj_l1pt_2",
    "TrigObj_l2pt",
    "TrigObj_eta",
    "TrigObj_phi",
    "TrigObj_id",
    "TrigObj_l1iso",
    "TrigObj_l1charge",
    "TrigObj_filterBits",
]
ele_branches = [
    "nElectron",
    "Electron_pt",
    "Electron_eta",
    "Electron_mvaFall17V2Iso_WP90",
]
other_branches = [
    "MET_pt",
    "HTXS_stage1_1_fine_cat_pTjet30GeV",
    "fixedGridRhoFastjetAll",
    "nLHEScaleWeight",
    "nLHEPdfWeight",
    "LHEPdfWeight",
    "HTXS_Higgs_pt",
    "HTXS_njets30",
]
event_flags = [
    "Flag_BadPFMuonFilter",
    "Flag_EcalDeadCellTriggerPrimitiveFilter",
    "Flag_HBHENoiseFilter",
    "Flag_HBHENoiseIsoFilter",
    "Flag_globalSuperTightHalo2016Filter",
    "Flag_goodVertices",
    "Flag_BadChargedCandidateFilter",
]

branches_2016 = [
    "HLT_IsoMu24",
    "HLT_IsoTkMu24",
    "L1PreFiringWeight_Nom",
    "L1PreFiringWeight_Up",
    "L1PreFiringWeight_Dn",
]
branches_2017 = [
    "HLT_IsoMu27",
    "L1PreFiringWeight_Nom",
    "L1PreFiringWeight_Up",
    "L1PreFiringWeight_Dn",
]
branches_2018 = ["HLT_IsoMu24"]

proc_columns = (
    event_branches
    + muon_branches
    + fsr_branches
    + jet_branches
    + genjet_branches
    + sajet_branches
    + vtx_branches
    + genpart_branches
    + trigobj_branches
    + ele_branches
    + other_branches
    + event_flags
)

branches = {
    "2016": proc_columns + branches_2016,
    "2017": proc_columns + branches_2017,
    "2018": proc_columns + branches_2018,
}
