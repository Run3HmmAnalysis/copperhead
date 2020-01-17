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

parameters.update({
    "n_pv" : for_all_years(0),
    "event_flags" : for_all_years(['BadPFMuonFilter','EcalDeadCellTriggerPrimitiveFilter',\
                                   'HBHENoiseFilter','HBHENoiseIsoFilter','globalSuperTightHalo2016Filter',\
                                   'goodVertices','BadChargedCandidateFilter']),
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
    "n_electrons" : for_all_years(0),
    
    "jet_pt_cut" : for_all_years(20.),
    "jet_eta_cut" : for_all_years(2.4),
    "jet_id" : for_all_years("mediumId"),
    
    "min_dr_mu_jet": for_all_years(0.4),
    
    "btag_loose_wp": {"2016": 0.2217, "2017": 0.1522, "2018": 0.1241},    
    "btag_medium_wp": {"2016": 0.6321, "2017": 0.4941, "2018": 0.4184},
    
})

    
    
    
    
    
    







