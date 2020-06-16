import time
import os,sys,glob
import argparse
from python.postprocessing import postprocess, plot, save_yields, load_yields, save_shapes, make_datacards, dnn_rebin, dnn_training
from config.variables import variables
from config.datasets import datasets
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", dest="year", default=2016, action='store')
parser.add_argument("-l", "--label", dest="label", default="apr23", action='store')
parser.add_argument("-dnn", "--dnn", action='store_true')
parser.add_argument("--rebin", action='store_true')
parser.add_argument("-t", "--train", action='store_true')
parser.add_argument("-js", "--jetsyst", action='store_true')
parser.add_argument("-i", "--iterative", action='store_true')
args = parser.parse_args()

#to_plot = ['dimuon_mass', 'dimuon_pt', 'mu1_pt', 'jet1_pt', 'jet1_eta', 'jet2_pt', 'jet2_eta', 'dnn_score']
to_plot = ['dimuon_mass', 'dnn_score']

vars_to_plot = {v.name:v for v in variables if v.name in to_plot}
#vars_to_plot = {v.name:v for v in variables}

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
    'ewk_lljj_mll105_160_ptj0','ewk_lljj_mll105_160','ewk_lljj_mll105_160_py',
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
    'vbf_powhegPS','vbf_powheg_herwig','vbf_powheg_dipole'
]

training_samples = {
    'background': ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160_ptj0'],#, 'ttjets_dl'],
    'signal': ['ggh_amcPS','vbf_powhegPS', 'vbf_powheg_herwig'],
}

if int(args.train)+int(args.dnn)+int(args.rebin)>1:
    print("Please specify only one of the options [train, dnn, rebin].")
    sys.exit()

if args.train:
    modules = ['to_pandas']
elif args.dnn:
    modules = ['to_pandas', 'dnn_evaluation', 'get_hists']
elif args.rebin:
    modules = ['to_pandas', 'dnn_evaluation']
else:
    modules =  ['to_pandas',  'get_hists']        

syst_variations = [os.path.basename(x) for x in glob.glob(f'/depot/cms/hmm/coffea/{args.year}_{args.label}/*') \
       if ('binned' not in x) and ('unbinned' not in x) ]


postproc_args = {
    'modules': modules,
    'year': args.year,
    'label': args.label,
    'in_path': f'/depot/cms/hmm/coffea/{args.year}_{args.label}/',
    'syst_variations': ['nominal']+syst_variations,
    'out_path': 'plots_new/',
    'samples':samples,
    'training_samples': training_samples,
    'channels': ['vbf','vbf_01j','vbf_2j'],
    'channel_groups': {'vbf':['vbf','vbf_01j','vbf_2j']},
    'regions': ['h-peak', 'h-sidebands'],
    'vars_to_plot': list(vars_to_plot.values()),
    'wgt_variations': True,
    'train_dnn': args.train,
    'rebin_dnn': args.rebin,
    'do_jetsyst': args.jetsyst
}

print(f"Start!")
tstart = time.time() 
dfs, hist_dfs, edges = postprocess(postproc_args, not args.iterative)

if args.train:
    print(f"Concatenating dataframes...")
    df = pd.concat(dfs)
    print("Starting DNN training...")
    dnn_training(df, postproc_args)
    print("DNN training complete!")
    sys.exit()

if args.rebin:
    print("Rebinning DNN...")
    boundaries = dnn_rebin(dfs, postproc_args)
    print(args.year, args.label, boundaries)
    sys.exit()

hist = {}
for var, hists in hist_dfs.items():
    print(f"Concatenating histograms: {var}")
    hist[var] = pd.concat(hists, ignore_index=True)

myvar = 'dnn_score' 

#print(f"Saving yields...")
#save_yields(vars_to_plot['dimuon_mass'], hist, edges[myvar], postproc_args)

print(f"Saving shapes...")
save_shapes(vars_to_plot[myvar], hist, edges[myvar], postproc_args)
print(f"Preparing datacards...")
make_datacards(vars_to_plot[myvar], hist, postproc_args)
   
for vname, var in vars_to_plot.items():
    print(f"Plotting: {vname}")
    for r in postproc_args['regions']:
        plot(var, hist, edges[vname], postproc_args, r)

elapsed = time.time() - tstart
print(f"Total time: {elapsed} s")
