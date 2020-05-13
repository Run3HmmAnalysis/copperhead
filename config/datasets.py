datasets = {
    "2016": {
#        "data_B": "/store/data/Run2016B_ver1/SingleMuon/NANOAOD/Nano25Oct2019_ver1-v1/",
        "data_B": "/store/data/Run2016B_ver2/SingleMuon/NANOAOD/Nano25Oct2019_ver2-v1/",    
        "data_C": "/store/data/Run2016C/SingleMuon/NANOAOD/Nano25Oct2019-v1/",
        "data_D": "/store/data/Run2016D/SingleMuon/NANOAOD/Nano25Oct2019-v1/",
        "data_E": "/store/data/Run2016E/SingleMuon/NANOAOD/Nano25Oct2019-v1/",
        "data_F": "/store/data/Run2016F/SingleMuon/NANOAOD/Nano25Oct2019-v1/",
        "data_G": "/store/data/Run2016G/SingleMuon/NANOAOD/Nano25Oct2019-v1/",
        "data_H": "/store/data/Run2016H/SingleMuon/NANOAOD/Nano25Oct2019-v1/",

        #"dy": "/store/mc/RunIISummer16NanoAODv6/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/",
        "dy": "/store/mc/RunIISummer16NanoAODv6/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/",
        "dy_0j" : "/store/mc/RunIISummer16NanoAODv6/DYToLL_0J_13TeV-amcatnloFXFX-pythia8/",
        "dy_1j" : "/store/mc/RunIISummer16NanoAODv6/DYToLL_1J_13TeV-amcatnloFXFX-pythia8/",
        "dy_2j" : "/store/mc/RunIISummer16NanoAODv6/DYToLL_2J_13TeV-amcatnloFXFX-pythia8/",
        
        #"dy_m105_160_amc": "/store/mc/RunIISummer16NanoAODv6/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/",
        "dy_m105_160_amc": "/store/mc/RunIISummer16NanoAODv6/DYJetsToLL_M-105To160_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/",
        "dy_m105_160_vbf_amc" : "/store/mc/RunIISummer16NanoAODv6/DYJetsToLL_M-105To160_VBFFilter_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/",
        "dy_m105_160_mg" : "/store/mc/RunIISummer16NanoAODv6/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/",
        
        "dy_m105_160_vbf_mg" : "/store/mc/RunIISummer16NanoAODv6/DYJetsToLL_M-105To160_VBFFilter_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/",
        
        "ttjets_dl": "/store/mc/RunIISummer16NanoAODv6/TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/",
        "ttjets_sl": "/store/mc/RunIISummer16NanoAODv6/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/",
        "ttw" : "/store/mc/RunIISummer16NanoAODv6/TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8/",
        "ttz" : "/store/mc/RunIISummer16NanoAODv6/TTZToLLNuNu_M-10_TuneCUETP8M1_13TeV-amcatnlo-pythia8/",
        
        
        "ewk_lljj_mll50_mjj120": "/store/mc/RunIISummer16NanoAODv6/EWK_LLJJ_MLL-50_MJJ-120_13TeV-madgraph-herwigpp/",
        "ewk_lljj_mll105_160": "/store/mc/RunIISummer16NanoAODv6/EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneEEC5_13TeV-madgraph-herwigpp/",
        "ewk_lljj_mll105_160_py": "/store/mc/RunIISummer16NanoAODv6/EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8/",
        "ewk_lljj_mll105_160_ptj0": "/store/group/local/hmm/FSRNANO2016MCV8d/EWK_LLJJ_MLL_105-160_ptJ-0_SM_5f_LO_TuneEEC5_13TeV-madgraph-herwigpp/",
        
        "st_tw_top": "/store/mc/RunIISummer16NanoAODv6/ST_tW_top_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/",
        "st_tw_antitop": "/store/mc/RunIISummer16NanoAODv6/ST_tW_antitop_5f_NoFullyHadronicDecays_13TeV-powheg_TuneCUETP8M1/",

        "ww_2l2nu": "/store/mc/RunIISummer16NanoAODv6/WWTo2L2Nu_13TeV-powheg/",
        "wz_3lnu" : "/store/mc/RunIISummer16NanoAODv6/WZTo3LNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/",
        "wz_2l2q" : "/store/mc/RunIISummer16NanoAODv6/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/",
        "wz_1l1nu2q" : "/store/mc/RunIISummer16NanoAODv6/WZTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8/",
        "zz" : "/store/mc/RunIISummer16NanoAODv6/ZZ_TuneCUETP8M1_13TeV-pythia8/",
        
        "www": "/store/mc/RunIISummer16NanoAODv6/WWW_4F_DiLeptonFilter_TuneCUETP8M1_13TeV-amcatnlo-pythia8/",
        "wwz": "/store/mc/RunIISummer16NanoAODv6/WWZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/",
        "wzz": "/store/mc/RunIISummer16NanoAODv6/WZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/",
        "zzz": "/store/mc/RunIISummer16NanoAODv6/ZZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/",

        "ggh_amcPS": "/store/mc/RunIISummer16NanoAODv6/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/",
#        "ggh_amcPS": ""
        "ggh_amcPS_m120": "/store/mc/RunIISummer16NanoAODv6/GluGluHToMuMu_M120_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/",
        "ggh_amcPS_m130": "/store/mc/RunIISummer16NanoAODv6/GluGluHToMuMu_M130_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/",
        
        "ggh_amcPS_TuneCP5down" : "/store/mc/RunIISummer16NanoAODv6/GluGluHToMuMu_M125_TuneCP5down_PSweights_13TeV_amcatnloFXFX_pythia8/",
        "ggh_amcPS_TuneCP5up" : "/store/mc/RunIISummer16NanoAODv6/GluGluHToMuMu_M125_TuneCP5up_PSweights_13TeV_amcatnloFXFX_pythia8/",
        "ggh_powheg" : "/store/mc/RunIISummer16NanoAODv6/GluGlu_HToMuMu_M125_13TeV_powheg_pythia8/",
        "ggh_powhegPS" : "/store/mc/RunIISummer16NanoAODv6/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/",
#        "ggh_powhegPS": "/store/group/local/hmm/FSRNANO2016MCV8a/GluGlu_HToMuMu_M125_13TeV_powheg_pythia8/",
        
        "ggh_powhegPS_m120" : "/store/mc/RunIISummer16NanoAODv6/GluGluHToMuMu_M-120_TuneCP5_PSweights_13TeV_powheg_pythia8/", 
        "ggh_powhegPS_m130" : "/store/mc/RunIISummer16NanoAODv6/GluGluHToMuMu_M-130_TuneCP5_PSweights_13TeV_powheg_pythia8/", 
        
        "vbf_amcPS": "/store/mc/RunIISummer16NanoAODv6/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/",
#        "vbf_amcPS": "/store/group/local/hmm/FSRNANO2016MCV8a/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/",
        "vbf_amcPS_m120": "/store/mc/RunIISummer16NanoAODv6/VBFHToMuMu_M120_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/",
        "vbf_amcPS_m130": "/store/mc/RunIISummer16NanoAODv6/VBFHToMuMu_M130_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/",
        
        "vbf_amcPS_TuneCP5down" : "/store/mc/RunIISummer16NanoAODv6/VBFHToMuMu_M125_TuneCP5down_PSweights_13TeV_amcatnlo_pythia8/",
        "vbf_amcPS_TuneCP5up" : "/store/mc/RunIISummer16NanoAODv6/VBFHToMuMu_M125_TuneCP5up_PSweights_13TeV_amcatnlo_pythia8/",
        "vbf_powheg" : "/store/mc/RunIISummer16NanoAODv6/VBF_HToMuMu_M125_13TeV_powheg_pythia8/",
        "vbf_powhegPS" : "/store/mc/RunIISummer16NanoAODv6/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/",
        "vbf_powhegPS" : "/store/group/local/hmm/FSRNANO2016MCV8a/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/",
        "vbf_powhegPS_m120" : "/store/mc/RunIISummer16NanoAODv6/VBFHToMuMu_M-120_TuneCP5_PSweights_13TeV_powheg_pythia8/", 
        "vbf_powhegPS_m130" : "/store/mc/RunIISummer16NanoAODv6/VBFHToMuMu_M-130_TuneCP5_PSweights_13TeV_powheg_pythia8/",         
        "vbf_amc_herwig" :  "/store/mc/RunIISummer16NanoAODv6/VBFHToMuMu_M-125_TuneEEC5_13TeV-amcatnlo-herwigpp/",
        #"vbf_powheg_herwig" : "/store/mc/RunIISummer16NanoAODv6/VBFHToMuMu_M-125_TuneEEC5_13TeV-powheg-herwigpp/", 
        "vbf_powheg_herwig" : "/store/group/local/hmm/FSRNANO2016MCV8h_06May2020/VBFHToMuMu_M-125_TuneCP5_13TeV-powheg-herwigpp/",
        "vbf_powheg_dipole" : "/store/group/local/hmm/FSRNANO2016MCV8h_06May2020/VBFHToMuMu_M-125_TuneCP5_13TeV-powheg-pythia_dipole/",
    
        "wmh" : "/store/mc/RunIISummer16NanoAODv6/WminusH_HToMuMu_WToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/",
        "wph" : "/store/mc/RunIISummer16NanoAODv6/WplusH_HToMuMu_WToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/",
        "tth" : "/store/mc/RunIISummer16NanoAODv6/ttHToMuMu_M125_TuneCP5_PSweights_13TeV-powheg-pythia8/",
        "zh" : "/store/mc/RunIISummer16NanoAODv6/ZH_HToMuMu_ZToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/",
    },

    "2017": {
#        "data_B": "/store/data/Run2017B/SingleMuon/NANOAOD/Nano25Oct2019-v1/",    
#        "data_C": "/store/data/Run2017C/SingleMuon/NANOAOD/Nano25Oct2019-v1/",
#        "data_D": "/store/data/Run2017D/SingleMuon/NANOAOD/Nano25Oct2019-v1/",
#        "data_E": "/store/data/Run2017E/SingleMuon/NANOAOD/Nano25Oct2019-v1/",
#        "data_F": "/store/data/Run2017F/SingleMuon/NANOAOD/Nano25Oct2019-v1/",

        "ggh_amcPS": "/store/group/local/hmm/FSRmyNanoProdMc2017_NANOV8a//GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/",
        "ggh_powhegPS": "/store/group/local/hmm/FSRmyNanoProdMc2017_NANOV8a//GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/",
        "vbf_powhegPS": "/store/group/local/hmm/FSRmyNanoProdMc2017_NANOV8a/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/",
        #"vbf_powheg_herwig": "/store/group/local/hmm/FSRmyNanoProdMc2017_NANOV8a/VBFHToMuMu_M-125_TuneEEC5_13TeV-powheg-herwigpp/",
        "vbf_amcPS": "/store/group/local/hmm/FSRmyNanoProdMc2017_NANOV8a//VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/",
        "vbf_powheg_herwig":"/store/group/local/hmm/FSRmyNanoProdMc2017_NANOV8h_06May2020/VBFHToMuMu_M-125_TuneCP5_13TeV-powheg-herwig7_fixed/",
        "vbf_powheg_dipole":"/store/group/local/hmm/FSRmyNanoProdMc2017_NANOV8h_06May2020/VBFHToMuMu_M-125_TuneCP5_13TeV-powheg-pythia_dipole/",
        
#        "dy_0j": "/store/mc/RunIIFall17NanoAODv6/DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8/",
#        "dy_1j": "/store/mc/RunIIFall17NanoAODv6/DYJetsToLL_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8/",
#        "dy_2j": "/store/mc/RunIIFall17NanoAODv6/DYJetsToLL_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8/",
#        "dy_m105_160_amc": "/store/mc/RunIIFall17NanoAODv6/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/",
#        "dy_m105_160_vbf_amc": "/store/mc/RunIIFall17NanoAODv6/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/",
#        "dy_m105_160_mg": "/store/mc/RunIIFall17NanoAODv6/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/",
#        "dy_m105_160_vbf_mg": "/store/mc/RunIIFall17NanoAODv6/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_13TeV-madgraphMLM-pythia8/",
 
#        "ttjets_dl": "/store/mc/RunIIFall17NanoAODv6/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/",
#        "ttjets_sl": "/store/mc/RunIIFall17NanoAODv6/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/",
#        "ttw": "/store/mc/RunIIFall17NanoAODv6/TTWJetsToLNu_TuneCP5_PSweights_13TeV-amcatnloFXFX-madspin-pythia8/",
#        "ttz": "/store/mc/RunIIFall17NanoAODv6/TTZToLLNuNu_M-10_TuneCP5_PSweights_13TeV-amcatnlo-pythia8/",

#        "ewk_lljj_mll50_mjj120": "/store/mc/RunIIFall17NanoAODv6/EWK_LLJJ_MLL-50_MJJ-120_TuneCH3_PSweights_13TeV-madgraph-herwig7/",
#        "ewk_lljj_mll105_160": "/store/mc/RunIIFall17NanoAODv6/EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7/",
#        "ewk_lljj_mll105_160_ptj0" : "/store/mc/RunIIFall17NanoAODv6/EWK_LLJJ_MLL_105-160_pTJ-0_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7/",        

#        "ww_2l2nu": "/store/mc/RunIIFall17NanoAODv6/WWTo2L2Nu_NNPDF31_TuneCP5_13TeV-powheg-pythia8/",
#        "wz_3lnu": "/store/mc/RunIIFall17NanoAODv6/WZTo3LNu_13TeV-powheg-pythia8/",
#        "wz_2l2q": "/store/mc/RunIIFall17NanoAODv6/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/",        
#        "wz_1l1nu2q": "/store/mc/RunIIFall17NanoAODv6/WZTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8/",
#        "zz": "/store/mc/RunIIFall17NanoAODv6/ZZ_TuneCP5_13TeV-pythia8/",

#        "st_t_top": "/store/mc/RunIIFall17NanoAODv6/ST_t-channel_top_5f_TuneCP5_13TeV-powheg-pythia8/",
#        "st_t_antitop": "/store/mc/RunIIFall17NanoAODv6/ST_t-channel_antitop_5f_TuneCP5_PSweights_13TeV-powheg-pythia8/",
#        "st_tw_antitop": "/store/mc/RunIIFall17NanoAODv6/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/",
#        "st_tw_top": "/store/mc/RunIIFall17NanoAODv6/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/",

#        "www": "/store/mc/RunIIFall17NanoAODv6/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/",
#        "wwz": "/store/mc/RunIIFall17NanoAODv6/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/",
#        "wzz": "/store/mc/RunIIFall17NanoAODv6/WZZ_TuneCP5_13TeV-amcatnlo-pythia8/",
#        "zzz": "/store/mc/RunIIFall17NanoAODv6/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/",

#        "wmh": "/store/mc/RunIIFall17NanoAODv6/WminusH_HToMuMu_WToAll_M125_13TeV_powheg_pythia8/",
#        "wph": "/store/mc/RunIIFall17NanoAODv6/WplusH_HToMuMu_WToAll_M125_13TeV_powheg_pythia8/",
#        "tth": "/store/mc/RunIIFall17NanoAODv6/ttHToMuMu_M125_TuneCP5_PSweights_13TeV-powheg-pythia8/",
#        "zh": "/store/mc/RunIIFall17NanoAODv6/ZH_HToMuMu_ZToAll_M125_13TeV_powheg_pythia8/",
        
        "data_B": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017B-31Mar2018-v1/",    
        "data_C": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017C-31Mar2018-v1/",
        "data_D": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017D-31Mar2018-v1/",
        "data_E": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017E-31Mar2018-v1/",
        "data_F": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdData2017_NANOV4/SingleMuon/RunIISummer16MiniAODv3_FSRmyNanoProdData2017_NANOV4_un2017F-31Mar2018-v1/",        
        
#        "ggh_amcPS": "",
#        "ggh_powhegPS": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/",
#        "vbf_powhegPS": "",
#        "vbf_amcPS": "",
        
        
        "dy_0j": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4b/DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8/",
        "dy_1j": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4b/DYJetsToLL_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8/",
        "dy_2j": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4b/DYJetsToLL_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8/",
        "dy_m105_160_amc": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4i/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/",
        "dy_m105_160_vbf_amc": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4b/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/",
        "dy_m105_160_mg": "",
        "dy_m105_160_vbf_mg": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4b/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_13TeV-madgraphMLM-pythia8/",
        
        "ttjets_dl": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4a/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/",
        "ttjets_sl": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4c/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/",
        "ttw": "",
        "ttz": "",
        
        "ewk_lljj_mll50_mjj120": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4a/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_PSweights_13TeV-madgraph-pythia8/",
        "ewk_lljj_mll105_160": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4a/EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7/",
        "ewk_lljj_mll105_160_py": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4a/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_PSweights_13TeV-madgraph-pythia8/",        
        "ewk_lljj_mll105_160_ptj0" : "/store/group/local/hmm/FSRmyNanoProdMc2017_NANOV8d/EWK_LLJJ_MLL_105-160_ptJ-0_SM_5f_LO_TuneEEC5_13TeV-madgraph-h7/",
        
        "ww_2l2nu": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4a/WWTo2L2Nu_NNPDF31_TuneCP5_PSweights_13TeV-powheg-pythia8/",
        "wz_3lnu": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4a/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/",
        "wz_2l2q": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4a/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/",

        "wz_1l1nu2q": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4a/WZTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8/",
        "zz": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4a/ZZ_TuneCP5_13TeV-pythia8/",
        
        "st_t_top": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4a/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8/",
        "st_t_antitop": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4a/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8/",
        "st_tw_antitop": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8/",
        "st_tw_top": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4a/ST_tW_top_5f_inclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8/",
        
        "wmh": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4/WminusH_HToMuMu_WToAll_M125_13TeV_powheg_pythia8/",
        "wph": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4a/WplusH_HToMuMu_WToAll_M125_13TeV_powheg_pythia8/",
        "tth": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4/ttHToMuMu_M125_TuneCP5_PSweights_13TeV-powheg-pythia8/",
        "zh": "/store/group/local/hmm/nanoAODv6_private/FSRmyNanoProdMc2017_NANOV4a/ZH_HToMuMu_ZToAll_M125_13TeV_powheg_pythia8/",
        
        "www": "",
        "wwz": "",
        "wzz": "",
        "zzz": "",
        
        
    },

    "2018": {
        #"data_A": "/store/data/Run2018A/SingleMuon/NANOAOD/Nano25Oct2019-v1/",    
        #"data_B": "/store/data/Run2018B/SingleMuon/NANOAOD/Nano25Oct2019-v1/",    
        #"data_C": "/store/data/Run2018C/SingleMuon/NANOAOD/Nano25Oct2019-v1/",
        #"data_D": "/store/data/Run2018D/SingleMuon/NANOAOD/Nano25Oct2019-v1/",

        #Samples with GeoFit
        "data_A": "/store/group/local/hmm/FSRmyNanoProdData2018ABC_NANOV8a/SingleMuon/RunIIData17_FSRmyNanoProdData2018ABC_NANOV8a_Run2018A-17Sep2018-v2/",
        "data_B": "/mnt/hadoop/store/group/local/hmm/FSRmyNanoProdData2018ABC_NANOV8a/SingleMuon/RunIIData17_FSRmyNanoProdData2018ABC_NANOV8a_Run2018B-17Sep2018-v1/",
        "data_C": "/mnt/hadoop/store/group/local/hmm/FSRmyNanoProdData2018ABC_NANOV8a/SingleMuon/RunIIData17_FSRmyNanoProdData2018ABC_NANOV8a_Run2018C-17Sep2018-v1/",
        "data_D": "/store/group/local/hmm/FSRmyNanoProdData2018D_NANOV8a/SingleMuon/RunIIData17_FSRmyNanoProdData2018D_NANOV8a_Run2018D-22Jan2019-v2/",
        
        "ggh_amcPS": "/store/mc/RunIIAutumn18NanoAODv6/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/",
        "ggh_amcPS": "/store/group/local/hmm/FSRmyNanoProdMc2018_NANOV8a/GluGluHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnloFXFX_pythia8/",
        "ggh_amcPS_TuneCP5down": "/store/mc/RunIIAutumn18NanoAODv6/GluGluHToMuMu_M125_TuneCP5down_PSweights_13TeV_amcatnloFXFX_pythia8/",
        "ggh_amcPS_TuneCP5up": "/store/mc/RunIIAutumn18NanoAODv6/GluGluHToMuMu_M125_TuneCP5up_PSweights_13TeV_amcatnloFXFX_pythia8/",
        "ggh_powhegPS": "/store/mc/RunIIAutumn18NanoAODv6/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/",
        "ggh_powhegPS": "/store/group/local/hmm/FSRmyNanoProdMc2018_NANOV8a/GluGluHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/",
        #"vbf_powhegPS": "/store/mc/RunIIAutumn18NanoAODv6/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/",
        "vbf_powhegPS": "/store/group/local/hmm/FSRmyNanoProdMc2018_NANOV8a/VBFHToMuMu_M-125_TuneCP5_PSweights_13TeV_powheg_pythia8/",

        "vbf_amcPS": "/store/mc/RunIIAutumn18NanoAODv6/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/",
        "vbf_amcPS": "/store/group/local/hmm/FSRmyNanoProdMc2018_NANOV8a/VBFHToMuMu_M125_TuneCP5_PSweights_13TeV_amcatnlo_pythia8/",

#        "vbf_powheg_herwig":"",
#        "vbf_powheg_dipole":"",
        
        "vbf_amcPS_TuneCP5down": "/store/mc/RunIIAutumn18NanoAODv6/VBFHToMuMu_M125_TuneCP5down_PSweights_13TeV_amcatnlo_pythia8/",
        "vbf_amcPS_TuneCP5up": "/store/mc/RunIIAutumn18NanoAODv6/VBFHToMuMu_M125_TuneCP5up_PSweights_13TeV_amcatnlo_pythia8/",
        "dy": "/store/mc/RunIIAutumn18NanoAODv6/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/",
        "dy_0j": "/store/mc/RunIIAutumn18NanoAODv6/DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8/",
        "dy_1j": "/store/mc/RunIIAutumn18NanoAODv6/DYJetsToLL_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8/",
        "dy_2j": "/store/mc/RunIIAutumn18NanoAODv6/DYJetsToLL_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8/",
        "dy_m105_160_amc": "/store/mc/RunIIAutumn18NanoAODv6/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/",
        "dy_m105_160_vbf_amc": "/store/mc/RunIIAutumn18NanoAODv6/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/",
        "dy_m105_160_mg": "/store/mc/RunIIAutumn18NanoAODv6/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/",
        "dy_m105_160_vbf_mg": "/store/mc/RunIIAutumn18NanoAODv6/DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/",
        "ttjets_dl": "/store/mc/RunIIAutumn18NanoAODv6/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/",
        "ttjets_sl": "/store/mc/RunIIAutumn18NanoAODv6/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/",
        "ttw": "/store/mc/RunIIAutumn18NanoAODv6/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/",
        "ttz": "/store/mc/RunIIAutumn18NanoAODv6/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/",
        "ewk_lljj_mll50_mjj120": "/store/mc/RunIIAutumn18NanoAODv6/EWK_LLJJ_MLL-50_MJJ-120_TuneCH3_PSweights_13TeV-madgraph-herwig7/",
        "ewk_lljj_mll105_160": "/store/mc/RunIIAutumn18NanoAODv6/EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7/",
        "ewk_lljj_mll105_160_py": "/store/mc/RunIIAutumn18NanoAODv6/EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCP5_PSweights_13TeV-madgraph-pythia8/", 
        "ewk_lljj_mll105_160_ptj0": "/store/group/local/hmm/FSRmyNanoProdMc2018_NANOV8d/EWK_LLJJ_MLL_105-160_ptJ-0_SM_5f_LO_TuneEEC5_13TeV-madgraph-h7/",
        "ww_2l2nu": "/store/mc/RunIIAutumn18NanoAODv6/WWTo2L2Nu_NNPDF31_TuneCP5_13TeV-powheg-pythia8/",
        "wz_3lnu": "/store/mc/RunIIAutumn18NanoAODv6/WZTo3LNu_TuneCP5_13TeV-powheg-pythia8/",
        "wz_2l2q": "/store/mc/RunIIAutumn18NanoAODv6/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/",
        "st_t_top": "/store/mc/RunIIAutumn18NanoAODv6/ST_t-channel_top_5f_TuneCP5_13TeV-powheg-pythia8/",
        "st_t_antitop": "/store/mc/RunIIAutumn18NanoAODv6/ST_t-channel_antitop_5f_TuneCP5_13TeV-powheg-pythia8/",
        "st_tw_antitop": "/store/mc/RunIIAutumn18NanoAODv6/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/",
        "st_tw_top": "/store/mc/RunIIAutumn18NanoAODv6/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/",
        "zz": "/store/mc/RunIIAutumn18NanoAODv6/ZZ_TuneCP5_13TeV-pythia8/",
        "wmh": "/store/mc/RunIIAutumn18NanoAODv6/WminusH_HToMuMu_WToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/",
        "wph": "/store/mc/RunIIAutumn18NanoAODv6/WplusH_HToMuMu_WToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/",
        "tth": "/store/mc/RunIIAutumn18NanoAODv6/ttHToMuMu_M125_TuneCP5_PSweights_13TeV-powheg-pythia8/",
        "zh": "/store/mc/RunIIAutumn18NanoAODv6/ZH_HToMuMu_ZToAll_M125_TuneCP5_PSweights_13TeV_powheg_pythia8/",
        "www": "/store/mc/RunIIAutumn18NanoAODv6/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/",
        "wwz": "/store/mc/RunIIAutumn18NanoAODv6/WWZ_TuneCP5_13TeV-amcatnlo-pythia8/",
        "wzz": "/store/mc/RunIIAutumn18NanoAODv6/WZZ_TuneCP5_13TeV-amcatnlo-pythia8/",
        "zzz": "/store/mc/RunIIAutumn18NanoAODv6/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/",
    }
}

all_dy = {
    "2016": ["dy", "dy_0j", "dy_1j", "dy_2j", "dy_m105_160_amc", "dy_m105_160_vbf_amc","dy_m105_160_mg", "dy_m105_160_vbf_mg"],
    "2017": ["dy", "dy_0j", "dy_1j", "dy_2j", "dy_m105_160_amc", "dy_m105_160_vbf_amc","dy_m105_160_mg", "dy_m105_160_vbf_mg"],
    "2018": ["dy", "dy_0j", "dy_1j", "dy_2j", "dy_m105_160_amc", "dy_m105_160_vbf_amc","dy_m105_160_mg", "dy_m105_160_vbf_mg"]
}

all_ewk = {
    "2016": ["ewk_lljj_mll50_mjj120", "ewk_lljj_mll105_160", "ewk_lljj_mll105_160_ptj0"],
    "2017": ["ewk_lljj_mll50_mjj120", "ewk_lljj_mll105_160", "ewk_lljj_mll105_160_ptj0"],
    "2018": ["ewk_lljj_mll50_mjj120", "ewk_lljj_mll105_160", "ewk_lljj_mll105_160_ptj0"]
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
    "2016": {'lumi': 35860., 'events': 804026710}, # to be verified
    "2017": {'lumi': 41900., 'events': 769080716}, # to be verified
    "2018": {'lumi': 59900., 'events': 985425574} # to be verified
}


