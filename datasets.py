datasets = {
    "2016": {
        "data_B": "/eos/cms/store/data/Run2016B_ver2/SingleMuon/NANOAOD/Nano25Oct2019_ver2-v1/*/",    
        "data_C": "/eos/cms/store/data/Run2016C/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",
        "data_D": "/eos/cms/store/data/Run2016D/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",
        "data_E": "/eos/cms/store/data/Run2016E/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",
        "data_F": "/eos/cms/store/data/Run2016F/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",
        "data_G": "/eos/cms/store/data/Run2016G/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",
        "data_H": "/eos/cms/store/data/Run2016H/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",
        
        "dy": "/eos/cms/store/mc/RunIISummer16NanoAODv6/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/*/*/",
         "dy_m105_160_amc": "/eos/cms/store/mc/RunIISummer16NanoAODv6/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/*/*/",
    
        "ttjets_dl": "/eos/cms/store/mc/RunIISummer16NanoAODv6/TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/NANOAODSIM/*/*/",

        "ewk_lljj_mll50_mjj120": "/eos/cms/store/mc/RunIISummer16NanoAODv6/EWK_LLJJ_MLL-50_MJJ-120_13TeV-madgraph-herwigpp/NANOAODSIM/*/*/",
        "ewk_lljj_mll105_160": "/eos/cms/store/mc/RunIISummer16NanoAODv6/EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneEEC5_13TeV-madgraph-herwigpp/NANOAODSIM/*/*/",
        "st_t_top": "",
        "st_t_antitop": "",
        "st_tw_top": "/eos/cms/store/mc/RunIISummer16NanoAODv6/ST_tW_top_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/NANOAODSIM/*/*/",
        "st_tw_antitop": "/eos/cms/store/mc/RunIISummer16NanoAODv6/ST_tW_antitop_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/NANOAODSIM/*/*/",

        "ww_2l2nu": "/eos/cms/store/mc/RunIISummer16NanoAODv6/WWTo2L2Nu_13TeV-powheg/NANOAODSIM/*/*/",
        "wz_3lnu": "/eos/cms/store/mc/RunIISummer16NanoAODv6/WZTo3LNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/*/*/",

        "www": "/eos/cms/store/mc/RunIISummer16NanoAODv6/WWW_4F_DiLeptonFilter_TuneCUETP8M1_13TeV-amcatnlo-pythia8/NANOAODSIM/*/*/",
        "wwz": "/eos/cms/store/mc/RunIISummer16NanoAODv6/WWZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/NANOAODSIM/*/*/",
        "wzz": "/eos/cms/store/mc/RunIISummer16NanoAODv6/WZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/NANOAODSIM/*/*/",
        "zzz": "/eos/cms/store/mc/RunIISummer16NanoAODv6/ZZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/NANOAODSIM/*/*/",

        "ggh_amcPS": "/eos/cms/store/mc/RunIISummer16NanoAODv6/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/NANOAODSIM/*/*/",
        "vbf_amcPS": "/eos/cms/store/mc/RunIISummer16NanoAODv6/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/NANOAODSIM/*/*/"
    
    }

    "2017": {
        "data_B": "/eos/cms/store/data/Run2017B/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",    
        "data_C": "/eos/cms/store/data/Run2017C/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",
        "data_D": "/eos/cms/store/data/Run2017D/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",
        "data_E": "/eos/cms/store/data/Run2017E/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",
        "data_F": "/eos/cms/store/data/Run2017F/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",
    
    }

    "2018": {
        "data_A": "/eos/cms/store/data/Run2018A/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",    
        "data_B": "/eos/cms/store/data/Run2018B/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",    
        "data_C": "/eos/cms/store/data/Run2018C/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",
        "data_D": "/eos/cms/store/data/Run2018D/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",
    #     "data_E": "/eos/cms/store/data/Run2018E/SingleMuon/NANOAOD/Nano25Oct2019-v1/*/",
    }
}


lumi_data = {
    "2016": {'lumi': 35860., 'events': 804026710}, # to be verified
    "2017": {'lumi': 41900., 'events': 769080716}, # to be verified
    "2018": {'lumi': 59900., 'events': 985425574} # to be verified
}