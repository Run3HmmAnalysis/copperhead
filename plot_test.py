from python.plotting import Plotter
from python.dimuon_processor import DimuonProcessor
from python.samples_info import SamplesInfo
import os

year = '2016'
suff = 'mar5_nojer'
vars_to_plot = []

vars_to_plot += ['dimuon_mass', 'dimuon_pt']
#vars_to_plot += ['dimuon_mass_res', 'dimuon_mass_res_rel']
#vars_to_plot += ['dimuon_eta', 'dimuon_phi', 'dimuon_cosThetaCS']
#vars_to_plot += ['dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR']

#vars_to_plot += ['mu1_pt', 'mu1_pt_over_mass', 'mu1_eta', 'mu1_phi', 'mu1_iso']
#vars_to_plot += ['mu2_pt', 'mu2_pt_over_mass', 'mu2_eta', 'mu2_phi', 'mu2_iso']

#vars_to_plot += ['jet1_pt']
#vars_to_plot += ['jet1_eta', 'jet1_phi', 'jet1_qgl']
#vars_to_plot += ['jet2_pt']
#vars_to_plot += ['jet2_eta', 'jet2_phi', 'jet2_qgl']
#vars_to_plot += ['deta_mumuj1', 'dphi_mumuj1']
#vars_to_plot += ['deta_mumuj2', 'dphi_mumuj2']
#vars_to_plot += ['min_deta_mumuj', 'min_dphi_mumuj']
#vars_to_plot += ['jj_mass']
#vars_to_plot += ['jj_pt', 'jj_eta', 'jj_phi']
#vars_to_plot += ['jj_dEta', 'jj_dPhi']

#vars_to_plot += ['mmjj_mass', 'mmjj_pt', 'mmjj_eta', 'mmjj_phi']
#vars_to_plot += ['zeppenfeld', 'rpt', 'nsoftjets2', 'nsoftjets5', 'htsoft2', 'htsoft5']
#vars_to_plot += ['njets', 'npv', 'met']

all_plots_pars = {
    'processor': DimuonProcessor(SamplesInfo(year)),
    'path': f'/depot/cms/hmm/coffea/all_{year}_{suff}/binned/',
#    'path': f'/depot/cms/hmm/coffea/all_feb19/',
    'chunked': True,
    'prefix': '',
    'samples': [
        'data_A', 'data_B','data_C','data_D','data_E','data_F','data_G','data_H',
         'dy_0j', 'dy_1j', 'dy_2j',
        'dy_m105_160_amc',
        'dy_m105_160_vbf_amc',
         'ewk_lljj_mll50_mjj120', 
         'ewk_lljj_mll105_160_ptj0',
        'ewk_lljj_mll105_160',
         'ttjets_dl', 
        'ggh_amcPS', 'vbf_powhegPS',
         'ttjets_sl', 'ttz', 'ttw',
         'st_tw_top','st_tw_antitop',
         'ww_2l2nu','w00z_2l2q','wz_3lnu','wz_1l1nu2q','zz',
        'www','wwz','wzz','zzz',
        ],
    'ewk_name': 'ewk_lljj_mll105_160',
    'vars': vars_to_plot,
    'year': year,
#    'rebin': 5,
    'regions' : ["z-peak", "h-sidebands", "h-peak"],
#     'channels': ["ggh_01j", "ggh_2j", "vbf"], 
#     'regions' : ["h-sidebands","h-peak"],
    'channels': ["vbf"],
#    'weights_by_ds': {'dy_0j':0.926, 'dy_1j':0.926, 'dy_2j':0.926, 'dy_m105_160_amc':0.996, 'dy_m105_160_vbf_amc':0.996}
}

try:
    os.mkdir(f'plots/test_{year}_{suff}/')
except:
    print("Output path already exists")
all_plots = Plotter(**all_plots_pars)
all_plots.make_datamc_comparison(do_inclusive=True, do_exclusive=True, normalize=False, logy=True, get_rates=False, save_to=f'plots/test_{year}_{suff}/', mc_factor=1)
