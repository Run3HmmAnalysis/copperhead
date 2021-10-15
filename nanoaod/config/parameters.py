def for_all_years(value):
    out = {k: value for k in ["2016", "2017", "2018"]}
    return out


parameters = {}

parameters["lumimask"] = {
    "2016": "data/lumimasks/Cert_271036-284044_13TeV_ReReco_07Aug2017_Collisions16_JSON.txt",
    "2017": "data/lumimasks/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt",
    "2018": "data/lumimasks/Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt",
}

parameters["hlt"] = {
    "2016": ["IsoMu24", "IsoTkMu24"],
    "2017": ["IsoMu27"],
    "2018": ["IsoMu24"],
}

parameters["roccor_file"] = {
    "2016": "data/roch_corr/RoccoR2016.txt",
    "2017": "data/roch_corr/RoccoR2017.txt",
    "2018": "data/roch_corr/RoccoR2018.txt",
}

parameters["nnlops_file"] = for_all_years("data/NNLOPS_reweight.root")

parameters["btag_sf_csv"] = {
    "2016": "data/btag/DeepCSV_2016LegacySF_V1.csv",
    "2017": "data/btag/DeepCSV_94XSF_V5_B_F.csv",
    "2018": "data/btag/DeepCSV_102XSF_V1.csv",
}

parameters["pu_file_data"] = {
    "2016": "data/pileup/PileupData_GoldenJSON_Full2016.root",
    "2017": "data/pileup/puData2017_withVar.root",
    "2018": "data/pileup/puData2018_withVar.root",
}

parameters["pu_file_mc"] = {
    "2016": "data/pileup/pileup_profile_Summer16.root",
    "2017": "data/pileup/mcPileup2017.root",
    "2018": "data/pileup/mcPileup2018.root",
}

# Reduced set
parameters["jec_unc_to_consider"] = {
    "2016": [
        "Absolute",
        "Absolute2016",
        "BBEC1",
        "BBEC12016",
        "EC2",
        "EC22016",
        "HF",
        "HF2016",
        "RelativeBal",
        "RelativeSample2016",
        "FlavorQCD",
    ],
    "2017": [
        "Absolute",
        "Absolute2017",
        "BBEC1",
        "BBEC12017",
        "EC2",
        "EC22017",
        "HF",
        "HF2017",
        "RelativeBal",
        "RelativeSample2017",
        "FlavorQCD",
    ],
    "2018": [
        "Absolute",
        "Absolute2018",
        "BBEC1",
        "BBEC12018",
        "EC2",
        "EC22018",
        "HF",
        "HF2018",
        "RelativeBal",
        "RelativeSample2018",
        "FlavorQCD",
    ],
}


def jec_variations(year):
    result = []
    for v in parameters["jec_unc_to_consider"][year]:
        result.append(v + "_up")
        result.append(v + "_down")
    return result


parameters["jec_variations"] = {
    year: jec_variations(year) for year in ["2016", "2017", "2018"]
}

jer_variations = [
    "jer1_up",
    "jer1_down",
    "jer2_up",
    "jer2_down",
    "jer3_up",
    "jer3_down",
    "jer4_up",
    "jer4_down",
    "jer5_up",
    "jer5_down",
    "jer6_up",
    "jer6_down",
]
parameters["jer_variations"] = for_all_years(jer_variations)

parameters["muSFFileList"] = {
    "2016": [
        {
            "id": (
                "data/muon_sf/year2016/RunBCDEF_SF_ID.root",
                "NUM_MediumID_DEN_genTracks_eta_pt",
            ),
            "iso": (
                "data/muon_sf/year2016/RunBCDEF_SF_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_eta_pt",
            ),
            "trig": (
                "data/muon_sf/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunBtoF.root",
                "IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 20.1 / 36.4,
        },
        {
            "id": (
                "data/muon_sf/year2016/RunGH_SF_ID.root",
                "NUM_MediumID_DEN_genTracks_eta_pt",
            ),
            "iso": (
                "data/muon_sf/year2016/RunGH_SF_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_eta_pt",
            ),
            "trig": (
                "data/muon_sf/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunGtoH.root",
                "IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu24_OR_IsoTkMu24_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 16.3 / 36.4,
        },
    ],
    "2017": [
        {
            "id": (
                "data/muon_sf/year2017/RunBCDEF_SF_ID.root",
                "NUM_MediumID_DEN_genTracks_pt_abseta",
            ),
            "iso": (
                "data/muon_sf/year2017/RunBCDEF_SF_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_pt_abseta",
            ),
            "trig": (
                "data/muon_sf/mu2017/EfficienciesAndSF_RunBtoF_Nov17Nov2017.root",
                "IsoMu27_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu27_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 1.0,
        }
    ],
    "2018": [
        {
            "id": (
                "data/muon_sf/year2018/RunABCD_SF_ID.root",
                "NUM_MediumID_DEN_genTracks_pt_abseta",
            ),
            "iso": (
                "data/muon_sf/year2018/RunABCD_SF_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_pt_abseta",
            ),
            "trig": (
                "data/muon_sf/mu2018/EfficienciesStudies_2018_trigger_EfficienciesAndSF_2018Data_BeforeMuonHLTUpdate.root",
                "IsoMu24_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu24_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 8.95 / 59.74,
        },
        {
            "id": (
                "data/muon_sf/year2018/RunABCD_SF_ID.root",
                "NUM_MediumID_DEN_genTracks_pt_abseta",
            ),
            "iso": (
                "data/muon_sf/year2018/RunABCD_SF_ISO.root",
                "NUM_TightRelIso_DEN_MediumID_pt_abseta",
            ),
            "trig": (
                "data/muon_sf/mu2018/EfficienciesStudies_2018_trigger_EfficienciesAndSF_2018Data_AfterMuonHLTUpdate.root",
                "IsoMu24_PtEtaBins/efficienciesDATA/abseta_pt_DATA",
                "IsoMu24_PtEtaBins/efficienciesMC/abseta_pt_MC",
            ),
            "scale": 50.79 / 59.74,
        },
    ],
}

parameters["zpt_weights_file"] = for_all_years("data/zpt/zpt_weights.histo.json")
parameters["puid_sf_file"] = for_all_years("data/puid_sf/PUIDMaps.root")
parameters["res_calib_path"] = for_all_years("data/res_calib/")

parameters.update(
    {
        "event_flags": for_all_years(
            [
                "BadPFMuonFilter",
                "EcalDeadCellTriggerPrimitiveFilter",
                "HBHENoiseFilter",
                "HBHENoiseIsoFilter",
                "globalSuperTightHalo2016Filter",
                "goodVertices",
                "BadChargedCandidateFilter",
            ]
        ),
        "do_l1prefiring_wgts": {"2016": True, "2017": True, "2018": False},
    }
)

parameters.update(
    {
        "muon_pt_cut": for_all_years(20.0),
        "muon_eta_cut": for_all_years(2.4),
        "muon_iso_cut": for_all_years(0.25),  # medium iso
        "muon_id": for_all_years("mediumId"),
        # "muon_flags": for_all_years(["isGlobal", "isTracker"]),
        "muon_flags": for_all_years([]),
        "muon_leading_pt": {"2016": 26.0, "2017": 29.0, "2018": 26.0},
        "muon_trigmatch_iso": for_all_years(0.15),  # tight iso
        "muon_trigmatch_dr": for_all_years(0.1),
        "muon_trigmatch_id": for_all_years("tightId"),
        "electron_pt_cut": for_all_years(20.0),
        "electron_eta_cut": for_all_years(2.5),
        "electron_id": for_all_years("mvaFall17V2Iso_WP90"),
        "jet_pt_cut": for_all_years(25.0),
        "jet_eta_cut": for_all_years(4.7),
        "jet_id": {"2016": "loose", "2017": "tight", "2018": "tight"},
        "jet_puid": {"2016": "loose", "2017": "2017corrected", "2018": "loose"},
        "min_dr_mu_jet": for_all_years(0.4),
        "btag_loose_wp": {"2016": 0.2217, "2017": 0.1522, "2018": 0.1241},
        "btag_medium_wp": {"2016": 0.6321, "2017": 0.4941, "2018": 0.4184},
        "softjet_dr2": for_all_years(0.16),
    }
)

parameters["n_pdf_variations"] = {"2016": 100, "2017": 33, "2018": 33}

parameters["dnn_max"] = {"2016": 1.75, "2017": 2.0, "2018": 2.35}

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
parameters["proc_columns"] = {
    "2016": proc_columns + branches_2016,
    "2017": proc_columns + branches_2017,
    "2018": proc_columns + branches_2018,
}
