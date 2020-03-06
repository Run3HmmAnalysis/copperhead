training_features = ['dimuon_mass', 'dimuon_pt', 'dimuon_eta', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR',\
                     'jj_mass', 'jj_eta', 'jj_phi', 'jj_pt', 'jj_dEta',\
                     'mmjj_mass', 'mmjj_eta', 'mmjj_phi','zeppenfeld',\
                     'jet1_pt', 'jet1_eta', 'jet1_qgl', 'jet2_pt', 'jet2_eta', 'jet2_qgl',\
                     'dimuon_cosThetaCS',\
                     'dimuon_mass_res_rel', 'deta_mumuj1', 'dphi_mumuj1', 'deta_mumuj2', 'dphi_mumuj2',\
                     'htsoft5',
                    ]

#jec_unc_to_consider = [
#            'AbsoluteMPFBias', 'AbsoluteScale', 'AbsoluteStat',
#            'FlavorQCD', 'Fragmentation',
#            'PileUpDataMC', 'PileUpPtBB', 'PileUpPtEC1', 'PileUpPtEC2', 'PileUpPtHF', 'PileUpPtRef',
#            'RelativeFSR', 'RelativeJEREC1', 'RelativeJEREC2', 'RelativeJERHF', 'RelativePtBB',
#            'RelativePtEC1', 'RelativePtEC2', 'RelativePtHF', 'RelativeBal', 'RelativeSample',
#            'RelativeStatEC', 'RelativeStatFSR', 'RelativeStatHF', 'SinglePionECAL', 'SinglePionHCAL', 'TimePtEta'
#        ]

# Reduced set
jec_unc_to_consider = [
            'Absolute', 'Absolute2016', 'Absolute2017', 'Absolute2018',
            'BBEC1', 'BBEC12016', 'BBEC12017', 'BBEC12018',
            'EC2', 'EC22016',  'EC22017', 'EC22018',
            'HF', 'HF2016', 'HF2017', 'HF2018',
            'RelativeBal', 
            'RelativeSample2016', 'RelativeSample2017', 'RelativeSample2018',   
            'FlavorQCD',   
        ]

def for_all_years(value):
    out = {k:value for k in ["2016", "2017", "2018"]}
    return out

parameters = {}

parameters["lumimask"] = {
    "2016": "data/lumimasks/Cert_271036-284044_13TeV_ReReco_07Aug2017_Collisions16_JSON.txt",
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

parameters["puLookup"] = {
    '2016': 'data/pileup/puLookup_2016.coffea',
    '2017': 'data/pileup/puLookup_2017.coffea',
    '2018': 'data/pileup/puLookup_2018.coffea',
}

parameters["puLookup_Up"] = {
    '2016': 'data/pileup/puLookup_Up_2016.coffea',
    '2017': 'data/pileup/puLookup_Up_2017.coffea',
    '2018': 'data/pileup/puLookup_Up_2018.coffea',
}

parameters["puLookup_Down"] = {
    '2016': 'data/pileup/puLookup_Down_2016.coffea',
    '2017': 'data/pileup/puLookup_Down_2017.coffea',
    '2018': 'data/pileup/puLookup_Down_2018.coffea',
}

parameters["muSFFileList"] = {'2016': [{'id'   :                                        ("data/muon_sf/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunBCDEF_SF_ID.root",\
                                                                                       "NUM_TightID_DEN_genTracks_eta_pt"),
                     'iso'   : ("data/muon_sf/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunBCDEF_SF_ISO.root",\
                                "NUM_TightRelIso_DEN_TightIDandIPCut_eta_pt"),
                     'trig'  : ("data/muon_sf/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunBtoF.root",\
                                "IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio"),
                     'scale' : 19.656062760/35.882515396},
                    {'id'     : ("data/muon_sf/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunGH_SF_ID.root",\
                                 "NUM_TightID_DEN_genTracks_eta_pt"),
                     'iso'   : ("data/muon_sf/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunGH_SF_ISO.root",\
                                "NUM_TightRelIso_DEN_TightIDandIPCut_eta_pt"),
                     'trig'  : ("data/muon_sf/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunGtoH.root",\
                                "IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio"),
                     'scale' : 16.226452636/35.882515396}],
             '2017': [{'id'     : ("data/muon_sf/mu2017/RunBCDEF_SF_ID.root", "NUM_TightID_DEN_genTracks_pt_abseta"),
                     'iso'   : ("data/muon_sf/mu2017/RunBCDEF_SF_ISO.root", "NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta"),
                     'trig'  : ("data/muon_sf/mu2017/EfficienciesAndSF_RunBtoF_Nov17Nov2017.root",\
                                "IsoMu27_PtEtaBins/abseta_pt_ratio"),
                     'scale' : 1.}],
             '2018': [{'id'     : ("data/muon_sf/mu2018/EfficienciesStudies_2018_rootfiles_RunABCD_SF_ID.root",\
                                 "NUM_TightID_DEN_TrackerMuons_pt_abseta"),
                     'iso'   : ("data/muon_sf/mu2018/EfficienciesStudies_2018_rootfiles_RunABCD_SF_ISO.root",\
                                "NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta"),
                     'trig'  : ("data/muon_sf/mu2018/EfficienciesStudies_2018_trigger_EfficienciesAndSF_2018Data_BeforeMuonHLTUpdate.root",\
                                "IsoMu24_PtEtaBins/abseta_pt_ratio"),
                     'scale' : 8.950818835/59.688059536},
                    {'id'     : ("data/muon_sf/mu2018/EfficienciesStudies_2018_rootfiles_RunABCD_SF_ID.root",\
                                 "NUM_TightID_DEN_TrackerMuons_pt_abseta"),
                     'iso'   : ("data/muon_sf/mu2018/EfficienciesStudies_2018_rootfiles_RunABCD_SF_ISO.root",\
                                "NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta"),
                     'trig'  : ("data/muon_sf/mu2018/EfficienciesStudies_2018_trigger_EfficienciesAndSF_2018Data_AfterMuonHLTUpdate.root",\
                                "IsoMu24_PtEtaBins/abseta_pt_ratio"),
                     'scale' : 50.737240701/59.688059536}],
            }


jec_levels = ['L1FastJet', 'L2Relative', 'L2L3Residual', 'L3Absolute']
jec_tag = {
    '2016': 'Summer16_07Aug2017_V11_MC',
    '2017': 'Fall17_17Nov2017_V32_MC',
    '2018': 'Autumn18_V19_MC'
}

parameters['jec_weight_sets'] = {
    '2016': [
                f"* * data/jec/Summer16_07Aug2017_V11_MC_L1FastJet_AK4PFchs.jec.txt",
                f"* * data/jec/Summer16_07Aug2017_V11_MC_L2Relative_AK4PFchs.jec.txt",
                f"* * data/jec/Summer16_07Aug2017_V11_MC_L2L3Residual_AK4PFchs.jec.txt",
                f"* * data/jec/Summer16_07Aug2017_V11_MC_L3Absolute_AK4PFchs.jec.txt",
                f"* * data/jec/Summer16_07Aug2017_V11_MC_Uncertainty_AK4PFchs.junc.txt",
                f"* * data/jec/Summer16_25nsV1_MC_PtResolution_AK4PFchs.jr.txt",
                f"* * data/jec/Summer16_25nsV1_MC_SF_AK4PFchs.jersf.txt",
                f"* * data/jec/Summer16_07Aug2017_V11_MC_UncertaintySources_AK4PFchs.junc.txt",       
                f"* * data/jec/Summer16_07Aug2017BCD_V11_DATA_L1FastJet_AK4PFchs.jec.txt",
                f"* * data/jec/Summer16_07Aug2017BCD_V11_DATA_L2Relative_AK4PFchs.jec.txt",
                f"* * data/jec/Summer16_07Aug2017BCD_V11_DATA_L2L3Residual_AK4PFchs.jec.txt",
                f"* * data/jec/Summer16_07Aug2017BCD_V11_DATA_L3Absolute_AK4PFchs.jec.txt",
                f"* * data/jec/Summer16_07Aug2017BCD_V11_DATA_UncertaintySources_AK4PFchs.junc.txt",
                f"* * data/jec/Summer16_07Aug2017BCD_V11_DATA_Uncertainty_AK4PFchs.junc.txt",
                f"* * data/jec/Summer16_07Aug2017EF_V11_DATA_L1FastJet_AK4PFchs.jec.txt",
                f"* * data/jec/Summer16_07Aug2017EF_V11_DATA_L2Relative_AK4PFchs.jec.txt",
                f"* * data/jec/Summer16_07Aug2017EF_V11_DATA_L2L3Residual_AK4PFchs.jec.txt",
                f"* * data/jec/Summer16_07Aug2017EF_V11_DATA_L3Absolute_AK4PFchs.jec.txt",
                f"* * data/jec/Summer16_07Aug2017EF_V11_DATA_UncertaintySources_AK4PFchs.junc.txt",
                f"* * data/jec/Summer16_07Aug2017EF_V11_DATA_Uncertainty_AK4PFchs.junc.txt",
                f"* * data/jec/Summer16_07Aug2017GH_V11_DATA_L1FastJet_AK4PFchs.jec.txt",
                f"* * data/jec/Summer16_07Aug2017GH_V11_DATA_L2Relative_AK4PFchs.jec.txt",
                f"* * data/jec/Summer16_07Aug2017GH_V11_DATA_L2L3Residual_AK4PFchs.jec.txt",
                f"* * data/jec/Summer16_07Aug2017GH_V11_DATA_L3Absolute_AK4PFchs.jec.txt",        
                f"* * data/jec/Summer16_07Aug2017GH_V11_DATA_UncertaintySources_AK4PFchs.junc.txt",
                f"* * data/jec/Summer16_07Aug2017GH_V11_DATA_Uncertainty_AK4PFchs.junc.txt",
            ],
    '2017': [
                f"* * data/jec/Fall17_17Nov2017_V32_MC_L1FastJet_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017_V32_MC_L2Relative_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017_V32_MC_Uncertainty_AK4PFchs.junc.txt",
                f"* * data/jec/Fall17_17Nov2017_V32_MC_L2L3Residual_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017_V32_MC_L3Absolute_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_V3_MC_PtResolution_AK4PFchs.jr.txt",
                f"* * data/jec/Fall17_V3_MC_SF_AK4PFchs.jersf.txt",
                f"* * data/jec/Fall17_17Nov2017_V32_MC_UncertaintySources_AK4PFchs.junc.txt",
                f"* * data/jec/Fall17_17Nov2017B_V32_DATA_L1FastJet_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017B_V32_DATA_L2Relative_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017B_V32_DATA_L2L3Residual_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017B_V32_DATA_L3Absolute_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017B_V32_DATA_UncertaintySources_AK4PFchs.junc.txt",
                f"* * data/jec/Fall17_17Nov2017B_V32_DATA_Uncertainty_AK4PFchs.junc.txt",
                f"* * data/jec/Fall17_17Nov2017C_V32_DATA_L1FastJet_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017C_V32_DATA_L2Relative_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017C_V32_DATA_L2L3Residual_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017C_V32_DATA_L3Absolute_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017C_V32_DATA_UncertaintySources_AK4PFchs.junc.txt",
                f"* * data/jec/Fall17_17Nov2017C_V32_DATA_Uncertainty_AK4PFchs.junc.txt",
                f"* * data/jec/Fall17_17Nov2017DE_V32_DATA_L1FastJet_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017DE_V32_DATA_L2Relative_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017DE_V32_DATA_L2L3Residual_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017DE_V32_DATA_L3Absolute_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017DE_V32_DATA_UncertaintySources_AK4PFchs.junc.txt",
                f"* * data/jec/Fall17_17Nov2017DE_V32_DATA_Uncertainty_AK4PFchs.junc.txt",
                f"* * data/jec/Fall17_17Nov2017F_V32_DATA_L1FastJet_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017F_V32_DATA_L2Relative_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017F_V32_DATA_L2L3Residual_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017F_V32_DATA_L3Absolute_AK4PFchs.jec.txt",
                f"* * data/jec/Fall17_17Nov2017F_V32_DATA_UncertaintySources_AK4PFchs.junc.txt",
                f"* * data/jec/Fall17_17Nov2017F_V32_DATA_Uncertainty_AK4PFchs.junc.txt",
                ],
    '2018': [
                f"* * data/jec/Autumn18_V19_MC_L1FastJet_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_V19_MC_L2Relative_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_V19_MC_L2L3Residual_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_V19_MC_L3Absolute_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_V19_MC_Uncertainty_AK4PFchs.junc.txt",
                f"* * data/jec/Autumn18_V7_MC_PtResolution_AK4PFchs.jr.txt",
                f"* * data/jec/Autumn18_V7_MC_SF_AK4PFchs.jersf.txt",
                f"* * data/jec/Autumn18_V19_MC_UncertaintySources_AK4PFchs.junc.txt",
                f"* * data/jec/Autumn18_RunA_V19_DATA_L1FastJet_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_RunA_V19_DATA_L2Relative_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_RunA_V19_DATA_L2L3Residual_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_RunA_V19_DATA_L3Absolute_AK4PFchs.jec.txt",        
                f"* * data/jec/Autumn18_RunA_V19_DATA_UncertaintySources_AK4PFchs.junc.txt",
                f"* * data/jec/Autumn18_RunA_V19_DATA_Uncertainty_AK4PFchs.junc.txt",
                f"* * data/jec/Autumn18_RunB_V19_DATA_L1FastJet_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_RunB_V19_DATA_L2Relative_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_RunB_V19_DATA_L2L3Residual_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_RunB_V19_DATA_L3Absolute_AK4PFchs.jec.txt",         
                f"* * data/jec/Autumn18_RunB_V19_DATA_UncertaintySources_AK4PFchs.junc.txt",
                f"* * data/jec/Autumn18_RunB_V19_DATA_Uncertainty_AK4PFchs.junc.txt",
                f"* * data/jec/Autumn18_RunC_V19_DATA_L1FastJet_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_RunC_V19_DATA_L2Relative_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_RunC_V19_DATA_L2L3Residual_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_RunC_V19_DATA_L3Absolute_AK4PFchs.jec.txt",         
                f"* * data/jec/Autumn18_RunC_V19_DATA_UncertaintySources_AK4PFchs.junc.txt",
                f"* * data/jec/Autumn18_RunC_V19_DATA_Uncertainty_AK4PFchs.junc.txt",
                f"* * data/jec/Autumn18_RunD_V19_DATA_L1FastJet_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_RunD_V19_DATA_L2Relative_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_RunD_V19_DATA_L2L3Residual_AK4PFchs.jec.txt",
                f"* * data/jec/Autumn18_RunD_V19_DATA_L3Absolute_AK4PFchs.jec.txt",         
                f"* * data/jec/Autumn18_RunD_V19_DATA_UncertaintySources_AK4PFchs.junc.txt",
                f"* * data/jec/Autumn18_RunD_V19_DATA_Uncertainty_AK4PFchs.junc.txt",
                ]
}

parameters['jec_names'] = {
    y: [f"{jec_tag[y]}_{level}_AK4PFchs" for level in jec_levels] for y in ['2016', '2017', '2018'] 
}

parameters['junc_names'] = {
    y: [f"{jec_tag[y]}_Uncertainty_AK4PFchs"] for y in ['2016', '2017', '2018']
}

parameters['jec_unc_sources'] = {
    y: f"{jec_tag[y]}_UncertaintySources_AK4PFchs" for y in ['2016', '2017', '2018']
}

parameters['jec_names_data'] = {
    '2016': {
        'B': [f"Summer16_07Aug2017BCD_V11_DATA_{level}_AK4PFchs" for level in jec_levels],
        'C': [f"Summer16_07Aug2017BCD_V11_DATA_{level}_AK4PFchs" for level in jec_levels],
        'D': [f"Summer16_07Aug2017BCD_V11_DATA_{level}_AK4PFchs" for level in jec_levels],
        'E': [f"Summer16_07Aug2017EF_V11_DATA_{level}_AK4PFchs" for level in jec_levels],
        'F': [f"Summer16_07Aug2017EF_V11_DATA_{level}_AK4PFchs" for level in jec_levels],
        'G': [f"Summer16_07Aug2017GH_V11_DATA_{level}_AK4PFchs" for level in jec_levels],
        'H': [f"Summer16_07Aug2017GH_V11_DATA_{level}_AK4PFchs" for level in jec_levels]
    },
    '2017': {
        'B': [f"Fall17_17Nov2017B_V32_DATA_{level}_AK4PFchs" for level in jec_levels],
        'C': [f"Fall17_17Nov2017C_V32_DATA_{level}_AK4PFchs" for level in jec_levels],
        'D': [f"Fall17_17Nov2017DE_V32_DATA_{level}_AK4PFchs" for level in jec_levels],
        'E': [f"Fall17_17Nov2017DE_V32_DATA_{level}_AK4PFchs" for level in jec_levels],
        'F': [f"Fall17_17Nov2017F_V32_DATA_{level}_AK4PFchs" for level in jec_levels],
    },
    '2018': {
        'A': [f"Autumn18_RunA_V19_DATA_{level}_AK4PFchs" for level in jec_levels],
        'B': [f"Autumn18_RunB_V19_DATA_{level}_AK4PFchs" for level in jec_levels],
        'C': [f"Autumn18_RunC_V19_DATA_{level}_AK4PFchs" for level in jec_levels],
        'D': [f"Autumn18_RunD_V19_DATA_{level}_AK4PFchs" for level in jec_levels],
    }
}


parameters['junc_names_data'] = {
    '2016': {
        'B': ["Summer16_07Aug2017BCD_V11_DATA_Uncertainty_AK4PFchs"],
        'C': ["Summer16_07Aug2017BCD_V11_DATA_Uncertainty_AK4PFchs"],
        'D': ["Summer16_07Aug2017BCD_V11_DATA_Uncertainty_AK4PFchs"],
        'E': ["Summer16_07Aug2017EF_V11_DATA_Uncertainty_AK4PFchs"],
        'F': ["Summer16_07Aug2017EF_V11_DATA_Uncertainty_AK4PFchs"],
        'G': ["Summer16_07Aug2017GH_V11_DATA_Uncertainty_AK4PFchs"],
        'H': ["Summer16_07Aug2017GH_V11_DATA_Uncertainty_AK4PFchs"]
    },
    '2017': {
        'B': ["Fall17_17Nov2017B_V32_DATA_Uncertainty_AK4PFchs"],
        'C': ["Fall17_17Nov2017C_V32_DATA_Uncertainty_AK4PFchs"],
        'D': ["Fall17_17Nov2017DE_V32_DATA_Uncertainty_AK4PFchs"],
        'E': ["Fall17_17Nov2017DE_V32_DATA_Uncertainty_AK4PFchs"],
        'F': ["Fall17_17Nov2017F_V32_DATA_Uncertainty_AK4PFchs"],
    },
    '2018': {
        'A': ["Autumn18_RunA_V19_DATA_Uncertainty_AK4PFchs"],
        'B': ["Autumn18_RunB_V19_DATA_Uncertainty_AK4PFchs"],
        'C': ["Autumn18_RunC_V19_DATA_Uncertainty_AK4PFchs"],
        'D': ["Autumn18_RunD_V19_DATA_Uncertainty_AK4PFchs"],
    }
}

parameters['jer_names'] = {
    '2016': ['Summer16_25nsV1_MC_PtResolution_AK4PFchs'],
    '2017': ['Fall17_V3_MC_PtResolution_AK4PFchs'],
    '2018': ['Autumn18_V7_MC_PtResolution_AK4PFchs'],
}

parameters['jersf_names'] = {
    '2016': ['Summer16_25nsV1_MC_SF_AK4PFchs'],
    '2017': ['Fall17_V3_MC_SF_AK4PFchs'],
    '2018': ['Autumn18_V7_MC_SF_AK4PFchs'],
}

parameters['jec_unc_names_data'] = {
    '2016': {
        'B': ["Summer16_07Aug2017BCD_V11_DATA_UncertaintySources_AK4PFchs"],
        'C': ["Summer16_07Aug2017BCD_V11_DATA_UncertaintySources_AK4PFchs"],
        'D': ["Summer16_07Aug2017BCD_V11_DATA_UncertaintySources_AK4PFchs"],
        'E': ["Summer16_07Aug2017EF_V11_DATA_UncertaintySources_AK4PFchs"],
        'F': ["Summer16_07Aug2017EF_V11_DATA_UncertaintySources_AK4PFchs"],
        'G': ["Summer16_07Aug2017GH_V11_DATA_UncertaintySources_AK4PFchs"],
        'H': ["Summer16_07Aug2017GH_V11_DATA_UncertaintySources_AK4PFchs"]
    },
    '2017': {
        'B': ["Fall17_17Nov2017B_V32_DATA_UncertaintySources_AK4PFchs"],
        'C': ["Fall17_17Nov2017C_V32_DATA_UncertaintySources_AK4PFchs"],
        'D': ["Fall17_17Nov2017DE_V32_DATA_UncertaintySources_AK4PFchs"],
        'E': ["Fall17_17Nov2017DE_V32_DATA_UncertaintySources_AK4PFchs"],
        'F': ["Fall17_17Nov2017F_V32_DATA_UncertaintySources_AK4PFchs"],
    },
    '2018': {
        'A': ["Autumn18_RunA_V19_DATA_UncertaintySources_AK4PFchs"],
        'B': ["Autumn18_RunB_V19_DATA_UncertaintySources_AK4PFchs"],
        'C': ["Autumn18_RunC_V19_DATA_UncertaintySources_AK4PFchs"],
        'D': ["Autumn18_RunD_V19_DATA_UncertaintySources_AK4PFchs"],
    }
}

parameters['zpt_weights_file'] = for_all_years("data/zpt/zpt_weights.histo.json")
parameters['puid_sf_file'] = for_all_years("data/puid_sf/PUIDMaps.root")
parameters['res_calib_path'] = for_all_years("data/res_calib/")

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
    
    "jet_pt_cut" : for_all_years(25.),
    "jet_eta_cut" : for_all_years(4.7),
    "jet_id" : {"2016": "loose", "2017": "tight", "2018": "tight"},
    "jet_puid" : {"2016": "loose", "2017": "2017corrected", "2018": "loose"},
    
    "min_dr_mu_jet": for_all_years(0.4),
    
    "btag_loose_wp": {"2016": 0.2217, "2017": 0.1522, "2018": 0.1241},    
    "btag_medium_wp": {"2016": 0.6321, "2017": 0.4941, "2018": 0.4184},    
    
    "softjet_dr2" : for_all_years(0.16),
})




event_branches = ['run', 'luminosityBlock', 'genWeight']
muon_branches = ['nMuon', 'Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_mass', 'Muon_charge', 'Muon_pfRelIso04_all']
jet_branches = ['nJet', 'Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass', 'Jet_qgl', 'Jet_jetId', 'Jet_puId']
vtx_branches = ['Pileup_nTrueInt', 'PV_npvsGood'] 
other_branches = ['MET_pt']
event_flags = ['Flag_BadPFMuonFilter','Flag_EcalDeadCellTriggerPrimitiveFilter',
                'Flag_HBHENoiseFilter','Flag_HBHENoiseIsoFilter',
                'Flag_globalSuperTightHalo2016Filter','Flag_goodVertices',
               #'Flag_BadChargedCandidateFilter'
              ]

branches_2016 = ['HLT_IsoMu24', 'HLT_IsoTkMu24', 'L1PreFiringWeight_Nom', 'L1PreFiringWeight_Up', 'L1PreFiringWeight_Dn']
branches_2017 = ['HLT_IsoMu27', 'L1PreFiringWeight_Nom', 'L1PreFiringWeight_Up', 'L1PreFiringWeight_Dn']
branches_2018 = ['HLT_IsoMu24']

proc_columns = event_branches + muon_branches + jet_branches + vtx_branches + other_branches + event_flags 
parameters["proc_columns"] = {
    "2016": proc_columns + branches_2016,
    "2017": proc_columns + branches_2017,
    "2018": proc_columns + branches_2018,
}
