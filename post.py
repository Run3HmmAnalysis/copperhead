import time
import os,sys,glob
import argparse
from python.postprocessing import postprocess, plot, save_yields, load_yields, save_shapes, make_datacards, dnn_rebin, dnn_training, prepare_root_files
from config.variables import variables
from config.datasets import datasets
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", dest="year", default=2016, action='store')
parser.add_argument("-l", "--label", dest="label", default="apr23", action='store')
parser.add_argument("-t", "--dnn_training", action='store_true')
parser.add_argument("-r", "--dnn_rebin", action='store_true')
parser.add_argument("-dnn", "--dnn_evaluation", action='store_true')
parser.add_argument("-dc", "--datacards", action='store_true')
parser.add_argument("-p", "--plot", action='store_true')
parser.add_argument("-js", "--jetsyst", action='store_true')
parser.add_argument("-i", "--iterative", action='store_true')
args = parser.parse_args()

if int(args.dnn_training)+int(args.dnn_rebin)+int(args.dnn_evaluation)+int(args.datacards)+int(args.plot)==0:
    print("Please specify option(s) to run:")
    print("-t --dnn_training")
    print("-r --dnn_rebin")
    print("-dnn --dnn_evaluation")
    print("-dc --datacards")
    print("-p --plot")
    sys.exit()

if args.dnn_training and (args.dnn_evaluation or args.datacards or args.plot):
    print("Can't combine 'training' option with 'evaluation','datacards' or 'plot'!")
    sys.exit()

if args.dnn_rebin and (args.dnn_evaluation or args.datacards or args.plot):
    print("Can't combine 'rebin' option with 'evaluation','datacards' or 'plot'!")
    sys.exit()

if args.dnn_evaluation and (args.dnn_training or args.dnn_rebin):
    print("Can't combine 'evaluation' option with 'training' or 'rebin'!")
    sys.exit()

if args.datacards and (args.dnn_training or args.dnn_rebin or args.plot):
    print("Can't combine 'datacards' option with 'training', 'rebin' or 'plot'!")
    sys.exit()

if args.plot and (args.dnn_training or args.dnn_rebin or args.datacards):
    print("Can't combine 'datacards' option with 'training', 'rebin' or 'datacards'!")
    sys.exit()

#to_plot = ['dimuon_mass', 'dimuon_pt', 'mu1_pt', 'jet1_pt', 'jet1_eta', 'jet2_pt', 'jet2_eta', 'dnn_score']
to_plot = ['dimuon_mass', 'dnn_score']

vars_to_plot = {v.name:v for v in variables if v.name in to_plot}
#vars_to_plot = {v.name:v for v in variables}

myvar = [v for v in variables if v.name == 'dnn_score'][0] # variable for templates and datacards

dnn_bins_all = {
    '2016':[0, 0.087, 0.34, 0.577, 0.787, 0.977, 1.152, 1.316, 1.475, 1.636, 1.801, 1.984, 2.44],
    '2017':[0, 0.061, 0.414, 0.694, 0.912, 1.093, 1.254, 1.402, 1.541, 1.672, 1.792, 1.906, 2.004, 2.102],
    '2018':[0, 0.005, 0.096, 0.216, 0.336, 0.45, 0.555, 0.652, 0.744, 0.834, 0.921, 1.007, 1.094, 1.184, 1.276, 1.374, 1.479, 1.591, 1.713, 1.859, 2.249]
    }
dnn_bins = dnn_bins_all[args.year]

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
    'ewk_lljj_mll105_160',
    'ewk_lljj_mll105_160_py',
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
    'vbf_powheg_herwig',
    'vbf_powheg_dipole'
]

training_samples = {
    'background': ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160_ptj0'],#, 'ttjets_dl'],
    'signal': ['ggh_amcPS','vbf_powhegPS', 'vbf_powheg_herwig'],
}
all_training_samples = []
for k,v in training_samples.items():
    all_training_samples.extend(v)

# keeping correct order just for debug output
def add_modules(modules, new_modules):
    for m in new_modules:
        if m not in modules: modules.append(m)
    return modules

load_unbinned_data = True
modules = []
options = []
if args.dnn_training:
    modules = add_modules(modules,['to_pandas'])
    samples = all_training_samples
    options += ['dnn_training']
if args.dnn_rebin:
    modules = add_modules(modules,['to_pandas', 'dnn_evaluation'])
    if not args.dnn_training: samples = ['vbf_powhegPS']
    options += ['dnn_rebin']
if args.dnn_evaluation:
    modules = add_modules(modules,['to_pandas', 'dnn_evaluation', 'get_hists'])
    options += ['dnn_evaluation']
    if not args.plot:
        vars_to_plot = {v.name:v for v in variables if v.name in ['dnn_score']}
if args.datacards:
    options += ['datacards']
    if not args.dnn_evaluation: load_unbinned_data = False
if args.plot:
    modules = add_modules(modules,['to_pandas',  'get_hists'])
    options += ['plot']

syst_variations = [os.path.basename(x) for x in glob.glob(f'/depot/cms/hmm/coffea/{args.year}_{args.label}/*') if ('nominal' not in x)]

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
    'train_dnn': args.dnn_training,
    'rebin_dnn': args.dnn_rebin,
    'do_jetsyst': args.jetsyst,
    'dnn_bins': dnn_bins
}

print(f"Start!")
print(f"Running options: {options}")
tstart = time.time() 

if load_unbinned_data:
    print(f"Will run modules: {modules}")
    print(f"Getting unbinned data (may take long)...")
    dfs, hist_dfs, edges = postprocess(postproc_args, not args.iterative)

    if args.dnn_training:
        print(f"Concatenating dataframes...")
        df = pd.concat(dfs)
        print("Starting DNN training...")
        dnn_training(df, postproc_args)
        print("DNN training complete!")
        sys.exit()

    if args.dnn_rebin:
        print("Rebinning DNN...")
        boundaries = dnn_rebin(dfs, postproc_args)
        print(args.year, args.label, boundaries)
        sys.exit()
    
    hist = {}
    for var, hists in hist_dfs.items():
        print(f"Concatenating histograms: {var}")
        hist[var] = pd.concat(hists, ignore_index=True)

    #print(f"Saving yields...")
    #save_yields(vars_to_plot['dimuon_mass'], hist, edges[myvar], postproc_args)
    if args.dnn_evaluation:
        print(f"Saving shapes: {myvar.name}")
        save_shapes(myvar, hist, dnn_bins, postproc_args)


if args.datacards:
    print(f"Preparing ROOT files with shapes ({myvar.name})...")    
    prepare_root_files(myvar, dnn_bins, postproc_args)
    print(f"Writing datacards...")
    make_datacards(myvar, postproc_args)

if args.plot:
    for vname, var in vars_to_plot.items():
        print(f"Plotting: {vname}")
        for r in postproc_args['regions']:
            plot(var, hist, edges[vname], postproc_args, r)

elapsed = time.time() - tstart
print(f"Total time: {elapsed} s")
