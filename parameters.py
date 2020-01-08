cross_sections = {
    "dy": 2075.14*3, # https://twiki.cern.ch/twiki/bin/viewauth/CMS/SummaryTable1G25ns; Pg 10: https://indico.cern.ch/event/746829/contributions/3138541/attachments/1717905/2772129/Drell-Yan_jets_crosssection.pdf 
    "dy_0j": 4620.52, #https://indico.cern.ch/event/673253/contributions/2756806/attachments/1541203/2416962/20171016_VJetsXsecsUpdate_PH-GEN.pdf
    "dy_1j": 859.59,
    "dy_2j": 338.26,
    "dy_m105_160_mg": 46.9479,
    "dy_m105_160_vbf_mg": 2.02,
    "dy_m105_160_amc": 46.9479, # https://docs.google.com/document/d/1bViX80nXQ_p-W4gI6Fqt9PNQ49B6cP1_FhcKwTZVujo/edit?usp=sharing
    "dy_m105_160_vbf_amc": 46.9479*0.0425242, #https://docs.google.com/document/d/1bViX80nXQ_p-W4gI6Fqt9PNQ49B6cP1_FhcKwTZVujo/edit?usp=sharing
    "ggh_powheg": 0.010571, #48.61 * 0.0002176; https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNHLHE2019
    "ggh_powhegPS": 0.010571,
    "ggh_amcPS": 0.010571,
    "ggh_amcPS_TuneCP5down": 0.010571,
    "ggh_amcPS_TuneCP5up": 0.010571,
    "ggh_amc": 0.010571,
    "vbf": 0.000823,
    "vbf_powheg_herwig": 0.000823,
    "vbf_powheg": 0.000823,
    "vbf_powhegPS": 0.000823,
    "vbf_amc_herwig": 0.000823,
    "vbf_amcPS_TuneCP5down": 0.000823,
    "vbf_amcPS_TuneCP5up": 0.000823,
    "vbf_amcPS": 0.000823,
    "vbf_amc": 0.000823,
    "wmh": 0.000116,
    "wph": 0.000183,
    "zh": 0.000192,
    "tth": 0.000110,
    "ttjets_dl": 85.656,
    "ttjets_sl": 687.0,
    "ww_2l2nu": 5.595,
    "wz_3lnu":  4.42965,
    "wz_2l2q": 5.595,
    "wz_1l1nu2q": 11.61,
    "zz": 16.523,
    "st_top": 136.02,
    "st_t_antitop": 80.95,
    "st_tw_top": 35.85,
    "st_tw_antitop": 35.85,
    "ewk_lljj_mll105_160": 0.0508896,

    # Note via Nan L.: the 2016 sample has a different tune, for which Stephane C.
    # computed a new cross-section from MINIAOD using
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/HowToGenXSecAnalyzer
    "ewk_lljj_mll50_mjj120": {"2016": 1.611, "2017": 1.700, "2018": 1.700},

    "ttw": 0.2001,
    "ttz": 0.2529,
    "st_t_top": 3.36,
    "www": 0.2086,
    "wwz": 0.1651,
    "wzz": 0.05565,
    "zzz": 0.01398
}