import os,glob
import argparse
from python.postprocessing import postprocess, plot, save_shapes
from config.variables import variables
from config.datasets import datasets
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", dest="year", default=2016, action='store')
parser.add_argument("-l", "--label", dest="label", default="apr23", action='store')
parser.add_argument("-o", "--out_path", dest="out_path", default="plots_new", action='store')
parser.add_argument("--dnn", action='store_true')
args = parser.parse_args()

to_plot = ['dimuon_mass']#, 'dimuon_pt', 'dnn_score']
vars_to_plot = {v.name:v for v in variables if v.name in to_plot}
#samples = list(datasets[args.year].keys())        
samples = [
    'data_A',
    'data_B',
    'data_C',
    'data_D',
    'data_E',
    'data_F',
    'data_G',
    'data_H',
    'dy_m105_160_amc',
    'dy_m105_160_vbf_amc',
    'ewk_lljj_mll105_160_ptj0',
    'ttjets_dl',
    'ttjets_sl',
    'ttz',
    'ttw',
    'st_tw_top','st_tw_antitop',
    'ww_2l2nu',
    'wz_2l2q',
    'wz_3lnu',
    'zz',
    'ggh_amcPS',
    'vbf_powhegPS',
]
#samples = [    'ggh_amcPS',
#    'vbf_powhegPS',]
if args.dnn:
    modules = ['to_pandas', 'dnn_evaluation', 'get_hists']
else:
    modules =  ['to_pandas',  'get_hists']        

syst_variations = [os.path.basename(x) for x in glob.glob(f'/depot/cms/hmm/coffea/all_{args.year}_{args.label}/*') \
       if ('binned' not in x) and ('unbinned' not in x) ]


postproc_args = {
    'modules': modules,
    'year': args.year,
    'label': args.label,
    'in_path': f'/depot/cms/hmm/coffea/all_{args.year}_{args.label}/',
    'syst_variations': ['nominal']+syst_variations,
    'out_path': args.out_path,
    'samples':samples,
    'channels': ['vbf'],
    'regions': ['h-peak', 'h-sidebands'],
    'vars_to_plot': list(vars_to_plot.values()),
}


dfs, hist_dfs, edges = postprocess(postproc_args)
hist = {}
for var, hists in hist_dfs.items():
    hist[var] = pd.concat(hists, ignore_index=True)

save_shapes(vars_to_plot['dimuon_mass'], hist, edges['dimuon_mass'], postproc_args)
    
for vname, var in vars_to_plot.items():
    for r in postproc_args['regions']:
        plot(var, hist, 'wgt_nominal', edges[vname], postproc_args, r)
#        plot(var, hist, 'wgt_puid_wgt_off', edges[vname], postproc_args, r)

    # inclusive
#    plot(var, hist, 'nominal', edges[vname], postproc_args)
