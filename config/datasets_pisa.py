datasets = {
    "2016": {
        "data_B": "",    
        "data_C": "",
        "data_D": "",
        "data_E": "",
        "data_F": "",
        "data_G": "",
        "data_H": "",

        "dy": "",
        "dy_0j" : "",
        "dy_1j" : "",
        "dy_2j" : "",
        
        "dy_m105_160_amc": "",
        "dy_m105_160_vbf_amc" : "",
        "dy_m105_160_mg" : "",
        
        "dy_m105_160_vbf_mg" : "",
        
        "ttjets_dl": "",
        "ttjets_sl": "",
        "ttw" : "",
        "ttz" : "",
        
        
        "ewk_lljj_mll50_mjj120": "",
        "ewk_lljj_mll105_160": "",

        "ewk_lljj_mll105_160_ptj0": "/store/user/arizzi/FSRNANO2016MCV8d/EWK_LLJJ_MLL_105-160_ptJ-0_SM_5f_LO_TuneEEC5_13TeV-madgraph-herwigpp/RunIISummer16MiniAODv3_FSRNANO2016MCV8d_bcad356ea4ed7e4f08b4",
        
        "st_tw_top": "",
        "st_tw_antitop": "",

        "ww_2l2nu": "",
        "wz_3lnu" : "",
        "wz_2l2q" : "",
        "wz_1l1nu2q" : "",
        "zz" : "",
        
        "www": "",
        "wwz": "",
        "wzz": "",
        "zzz": "",

        "ggh_amcPS": "/store/group/local/hmm/FSRNANO2016MCV8a/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/",
        "ggh_amcPS_TuneCP5down" : "/store/group/local/hmm/FSRNANO2016MCV8a/GluGluHToMuMu_M125_TuneCP5down_PSweights_13TeV_amcatnloFXFX_pythia8/",
        "ggh_amcPS_TuneCP5up" : "/store/group/local/hmm/FSRNANO2016MCV8a/GluGluHToMuMu_M125_TuneCP5up_PSweights_13TeV_amcatnloFXFX_pythia8/",
        "ggh_powheg" : "/mnt/hadoop/store/group/local/hmm/FSRNANO2016MCV8a/GluGlu_HToMuMu_M125_13TeV_powheg_pythia8/",
        "ggh_powhegPS" : "",        
        
        "vbf_amcPS": "/store/group/local/hmm/FSRNANO2016MCV8a/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/",
        "vbf_amcPS_TuneCP5down" : "",
        "vbf_amcPS_TuneCP5up" : "",
        "vbf_powheg" : "/store/group/local/hmm/FSRNANO2016MCV8a/VBF_HToMuMu_M125_13TeV_powheg_pythia8/",
        "vbf_powhegPS" : "/store/group/local/hmm/FSRNANO2016MCV8a/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/",        
        "vbf_amc_herwig" :  "/store/group/local/hmm/FSRNANO2016MCV8a/VBFHToMuMu_M-125_TuneEEC5_13TeV-amcatnlo-herwigpp/",
        "vbf_powheg_herwig" : "/store/group/local/hmm/FSRNANO2016MCV8a/VBFHToMuMu_M-125_TuneEEC5_13TeV-powheg-herwigpp/",      
    
        "wmh" : "",
        "wph" : "",
        "tth" : "",
        "zh" : "",
    },

    "2017": {

    },

    "2018": {

    }
}

all_dy = {
    "2016": ["dy", "dy_0j", "dy_1j", "dy_2j", "dy_m105_160_amc", "dy_m105_160_vbf_amc","dy_m105_160_mg", "dy_m105_160_vbf_mg"],
    "2017": ["dy", "dy_0j", "dy_1j", "dy_2j", "dy_m105_160_amc", "dy_m105_160_vbf_amc","dy_m105_160_mg", "dy_m105_160_vbf_mg"],
    "2018": ["dy", "dy_0j", "dy_1j", "dy_2j", "dy_m105_160_amc", "dy_m105_160_vbf_amc","dy_m105_160_mg", "dy_m105_160_vbf_mg"]
}

all_ewk = {
    "2016": ["ewk_lljj_mll50_mjj120", "ewk_lljj_mll105_160"],
    "2017": ["ewk_lljj_mll50_mjj120", "ewk_lljj_mll105_160", "ewk_lljj_mll105_160_ptj0"],
    "2018": ["ewk_lljj_mll50_mjj120", "ewk_lljj_mll105_160"]
}

all_ggh = {
    "2016": ["ggh_amcPS", "ggh_amcPS_TuneCP5down", "ggh_amcPS_TuneCP5up", "ggh_powheg", "ggh_powhegPS"],
    "2017": ["ggh_amc", "ggh_amcPS", "ggh_amcPS_TuneCP5down", "ggh_amcPS_TuneCP5up", "ggh_powheg", "ggh_powhegPS"],
    "2018": ["ggh_amcPS", "ggh_amcPS_TuneCP5down", "ggh_amcPS_TuneCP5up", "ggh_powhegPS"]
}

all_vbf = {
    "2016": ["vbf_amcPS", "vbf_amcPS_TuneCP5down", "vbf_amcPS_TuneCP5up", "vbf_powheg", "vbf_powhegPS", "vbf_amc_herwig", "vbf_powheg_herwig"],
    "2017": ["vbf_amc", "vbf_amcPS", "vbf_amcPS_TuneCP5down", "vbf_amcPS_TuneCP5up", "vbf_amc_herwig", "vbf_powheg_herwig"],
    "2018": ["vbf_amcPS", "vbf_amcPS_TuneCP5down", "vbf_amcPS_TuneCP5up", "vbf_powhegPS"]
}

lumi_data = {
    "2016": {'lumi': 35860., 'events': 670277010}, # to be verified
    "2017": {'lumi': 41900., 'events': 769080716}, # to be verified
    "2018": {'lumi': 59900., 'events': 985425574} # to be verified
}