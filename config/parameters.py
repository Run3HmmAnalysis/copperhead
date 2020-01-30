def for_all_years(value):
    out = {k:value for k in ["2016", "2017", "2018"]}
    return out

parameters = {}

parameters["lumimask"] = {
    "2016": "data/lumimasks/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt",
    "2017": "data/lumimasks/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt",
    "2018": "data/lumimasks/Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt"
}
parameters["hlt"] = {
    "2016": ['IsoMu24', 'IsoTkMu24'],
    "2017": ['IsoMu27'],
    "2018": ['IsoMu24']
}

parameters["roccor_file"] = {
    "2016": "data/roch_corr/RoccoR2016.txt",
    "2017": "data/roch_corr/RoccoR2017.txt",
    "2018": "data/roch_corr/RoccoR2018.txt",
}

# TODO: put correct links for 2017/18
parameters["muSFFileList"] = for_all_years(
         [{'id'   : ("data/muon_sf/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunBCDEF_SF_ID.root", "NUM_TightID_DEN_genTracks_eta_pt"),
          'iso'   : ("data/muon_sf/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunBCDEF_SF_ISO.root", "NUM_TightRelIso_DEN_TightIDandIPCut_eta_pt"),
          'trig'  : ("data/muon_sf/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunBtoF.root", "IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio"),
          'scale' : 19.656062760/35.882515396},
        {'id'     : ("data/muon_sf/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunGH_SF_ID.root", "NUM_TightID_DEN_genTracks_eta_pt"),
         'iso'   : ("data/muon_sf/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunGH_SF_ISO.root", "NUM_TightRelIso_DEN_TightIDandIPCut_eta_pt"),
         'trig'  : ("data/muon_sf/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunGtoH.root", "IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio"),
         'scale' : 16.226452636/35.882515396}]
)

parameters['zpt_weights_file'] = for_all_years("data/zpt/zpt_weights.histo.json")

parameters.update({
    "event_flags" : for_all_years(['BadPFMuonFilter','EcalDeadCellTriggerPrimitiveFilter',\
                                   'HBHENoiseFilter','HBHENoiseIsoFilter','globalSuperTightHalo2016Filter',\
                                   'goodVertices','BadChargedCandidateFilter']),
    "do_l1prefiring_wgts" : {"2016": True, "2017": True, "2018": False}
})

parameters.update({
    "muon_pt_cut" : for_all_years(20.),
    "muon_eta_cut" : for_all_years(2.4),
    "muon_iso_cut" : for_all_years(0.25), # medium iso
    "muon_id" : for_all_years("mediumId"),
    "muon_flags": for_all_years(["isGlobal", "isTracker"]),

    "muon_leading_pt": {"2016": 26., "2017": 29., "2018": 26.},
    "muon_trigmatch_iso" : for_all_years(0.15), # tight iso
    "muon_trigmatch_dr" : for_all_years(0.1),
    "muon_trigmatch_id" : for_all_years("tightId"),
    
    "electron_pt_cut" : for_all_years(20.),
    "electron_eta_cut" : for_all_years(2.5),
    "electron_id" : for_all_years("mvaFall17V2Iso_WP90"),
    
    "jet_pt_cut" : for_all_years(20.),
    "jet_eta_cut" : for_all_years(4.7),
    "jet_id" : {"2016": "loose", "2017": "tight", "2018": "tight"},
    "jet_puid" : {"2016": "loose", "2017": "2017corrected", "2018": "loose"},
    
    "min_dr_mu_jet": for_all_years(0.4),
    
    "btag_loose_wp": {"2016": 0.2217, "2017": 0.1522, "2018": 0.1241},    
    "btag_medium_wp": {"2016": 0.6321, "2017": 0.4941, "2018": 0.4184},    
})




event_branches = ['run', 'luminosityBlock', 'genWeight']
muon_branches = ['nMuon', 'Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_mass', 'Muon_charge', 'Muon_pfRelIso04_all']
jet_branches = ['nJet', 'Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass', 'Jet_qgl', 'Jet_jetId', 'Jet_puId']
vtx_branches = ['Pileup_nTrueInt', 'PV_npvsGood'] 
other_branches = ['MET_pt']
event_flags = ['Flag_BadPFMuonFilter','Flag_EcalDeadCellTriggerPrimitiveFilter',
                'Flag_HBHENoiseFilter','Flag_HBHENoiseIsoFilter',
                'Flag_globalSuperTightHalo2016Filter','Flag_goodVertices','Flag_BadChargedCandidateFilter']

branches_2016 = ['HLT_IsoMu24', 'HLT_IsoTkMu24', 'L1PreFiringWeight_Nom']
branches_2017 = ['HLT_IsoMu27', 'L1PreFiringWeight_Nom']
branches_2018 = ['HLT_IsoMu24', 'L1PreFiringWeight_Nom']

proc_columns = event_branches + muon_branches + jet_branches + vtx_branches + other_branches + event_flags 
parameters["proc_columns"] = {
    "2016": proc_columns + branches_2016,
    "2017": proc_columns + branches_2017,
    "2018": proc_columns + branches_2018,
}
