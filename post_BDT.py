import time
import os,sys,glob
import argparse
from python.postprocessing_BDT import postprocess, plot, save_yields, load_yields, save_shapes, make_datacards, dnn_rebin, classifier_train, prepare_root_files
from config.variables import variables
from config.datasets import datasets
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", dest="year", default='', action='store')
parser.add_argument("-l", "--label", dest="label", default="jun16", action='store')
parser.add_argument("-dnn", "--dnn", action='store_true')
parser.add_argument("-bdt", "--bdt", action='store_true')
parser.add_argument("-t", "--train", action='store_true')
parser.add_argument("-r", "--dnn_rebin", action='store_true')
parser.add_argument("-evaluation", "--evaluation", action='store_true')
parser.add_argument("-dc", "--datacards", action='store_true')
parser.add_argument("-p", "--plot", action='store_true')
parser.add_argument("-js", "--jetsyst", action='store_true')
parser.add_argument("-i", "--iterative", action='store_true')
args = parser.parse_args()

if int(args.train)+int(args.dnn_rebin)+int(args.evaluation)+int(args.datacards)+int(args.plot)==0:
    print("Please specify option(s) to run:")
    print("-t --train")
    print("-r --dnn_rebin")
    print("-dnn --evaluation")
    print("-dc --datacards")
    print("-p --plot")
    sys.exit()

if (args.year=='') and not args.train:
    print("Year must be specified! Merging data from different years is only allowed for training.")
    sys.exit()

if args.train and (args.evaluation or args.datacards or args.plot):
    print("Can't combine 'training' option with 'evaluation','datacards' or 'plot'!")
    sys.exit()

if args.dnn_rebin and (args.evaluation or args.datacards or args.plot):
    print("Can't combine 'rebin' option with 'evaluation','datacards' or 'plot'!")
    sys.exit()

if args.evaluation and (args.train or args.dnn_rebin):
    print("Can't combine 'evaluation' option with 'training' or 'rebin'!")
    sys.exit()

if args.datacards and (args.train or args.dnn_rebin or args.plot):
    print("Can't combine 'datacards' option with 'training', 'rebin' or 'plot'!")
    sys.exit()

if args.plot and (args.train or args.dnn_rebin or args.datacards):
    print("Can't combine 'datacards' option with 'training', 'rebin' or 'datacards'!")
    sys.exit()

to_plot = ['dimuon_mass', 'dimuon_pt', 'mu1_pt', 'jet1_pt', 'jet1_eta', 'jet2_pt', 'jet2_eta', 'dnn_score']
#to_plot = ['dimuon_mass', 'dnn_score']

vars_to_plot = {v.name:v for v in variables if v.name in to_plot}
#vars_to_plot = {v.name:v for v in variables}
if(args.dnn):
    myvar = [v for v in variables if v.name == 'dnn_score'][0] # variable for templates and datacards
if(args.bdt):
    myvar = [v for v in variables if v.name == 'bdt_score'][0] # variable for templates and datacards

# old
#dnn_bins_all = {
#    '2016': [0, 0.188, 0.371, 0.548, 0.718, 0.883, 1.043, 1.196, 1.343, 1.485, 1.62, 1.75, 1.874, 2.3],
#    '2017': [0, 0.23, 0.449, 0.657, 0.853, 1.038, 1.211, 1.373, 1.523, 1.662, 1.789, 1.905, 2.01, 2.3],
#    '2018': [0, 0.22, 0.43, 0.63, 0.82, 0.999, 1.168, 1.327, 1.476, 1.614, 1.742, 1.86, 1.968, 2.3]
#    }

# old variables, new binning
dnn_bins_all_ = {
    '2016': [0, 0.152, 0.368, 0.571, 0.759, 0.932, 1.096, 1.25, 1.396, 1.534, 1.661, 1.775, 1.873, 2.3],
    '2017': [0, 0.295, 0.593, 0.83, 1.017, 1.181, 1.329, 1.464, 1.593, 1.712, 1.824, 1.926, 2.012, 2.3],
    '2018': [0, 0.065, 0.447, 0.735, 0.958, 1.147, 1.318, 1.473, 1.614, 1.736, 1.837, 1.915, 1.967, 2.3]
    }

# new variables, new binning
dnn_bins_all_ = {
    '2016': [0, 0.172, 0.405, 0.613, 0.8, 0.973, 1.138, 1.29, 1.434, 1.573, 1.704, 1.818, 1.919, 2.3],
    '2017': [0, 0.312, 0.622, 0.86, 1.051, 1.219, 1.373, 1.514, 1.649, 1.774, 1.888, 1.989, 2.086, 2.3],
    '2018': [0, 0.064, 0.425, 0.705, 0.928, 1.12, 1.291, 1.445, 1.588, 1.717, 1.831, 1.916, 1.964, 2.3]
    }

# new variables, new binning, DNN trained with mix of all years, using event weights and input samples sync w/ Pisa
dnn_bins_all = {
    '2016': [0, 0.301, 0.675, 0.974, 1.224, 1.45, 1.657, 1.846, 2.022, 2.187, 2.335, 2.438, 2.502, 2.8],
    '2017': [0, 0.333, 0.706, 1.006, 1.257, 1.482,1.685, 1.868, 2.043, 2.204, 2.349, 2.443, 2.505, 2.8],
    '2018': [0, 0.108, 0.7,   1.104, 1.416, 1.67, 1.879, 2.059, 2.219, 2.35,  2.435, 2.483, 2.513, 2.8]
    }

dnn_bins = dnn_bins_all[args.year] if args.year else []

samples_ = [
    'ggh_amcPS'
]
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
    'ewk_lljj_mll105_160_py_dipole',
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
    'background': ['dy_m105_160_mg', 'dy_m105_160_vbf_mg', 'ewk_lljj_mll105_160_ptj0', 'ttjets_dl', 'ttjets_sl'],
    'signal': ['vbf_powhegPS'],
}
all_training_samples = []
for k,v in training_samples.items():
    all_training_samples.extend(v)
    
pt_variations = []
pt_variations += ['nominal']
pt_variations += ['Absolute', f'Absolute{args.year}']
pt_variations += ['BBEC1', f'BBEC1{args.year}']
pt_variations += ['EC2', f'EC2{args.year}']
pt_variations += ['FlavorQCD']
pt_variations += ['HF',f'HF{args.year}']
pt_variations += ['RelativeBal', f'RelativeSample{args.year}']
pt_variations += ['jer1','jer2','jer3','jer4','jer5','jer6']

all_pt_variations = []
for ptvar in pt_variations:
    if ptvar=='nominal':
        all_pt_variations += ['nominal']
    else:
        all_pt_variations += [f'{ptvar}_up']
        all_pt_variations += [f'{ptvar}_down'] 

if (not args.jetsyst) or args.train or args.dnn_rebin:
    all_pt_variations = ['nominal']

# keeping correct order just for debug output
def add_modules(modules, new_modules):
    for m in new_modules:
        if m not in modules: modules.append(m)
    return modules

load_unbinned_data = True
modules = []
options = []
if args.train:
    modules = add_modules(modules,['to_pandas'])
    samples = all_training_samples
    options += ['train']
if args.dnn_rebin:
    modules = add_modules(modules,['to_pandas', 'evaluation'])
    if not args.train: samples = ['vbf_powheg_dipole', 'ggh_amcPS']
    options += ['dnn_rebin']
if args.evaluation:
    modules = add_modules(modules,['to_pandas', 'evaluation', 'get_hists'])
    options += ['evaluation']
    if not args.plot:
        if args.dnn:
            vars_to_plot = {v.name:v for v in variables if v.name in ['dnn_score']}
        if args.bdt:
            vars_to_plot = {v.name:v for v in variables if v.name in ['bdt_score']}
if args.datacards:
    options += ['datacards']
    if not args.evaluation: load_unbinned_data = False
if args.plot:
    modules = add_modules(modules,['to_pandas',  'get_hists'])
    options += ['plot']


postproc_args = {
    'modules': modules,
    'year': args.year,
    'label': args.label,
    'in_path': f'/depot/cms/hmm/coffea/',
    'evaluate_allyears_dnn': False,
    'syst_variations': all_pt_variations,
    'out_path': 'plots_new/',
    'samples':samples,
    'training_samples': training_samples,
    'channels': ['vbf','vbf_01j','vbf_2j'],
    'channel_groups': {'vbf':['vbf','vbf_01j','vbf_2j']},
    'regions': ['h-peak', 'h-sidebands'],
    'vars_to_plot': list(vars_to_plot.values()),
    'wgt_variations': True,
    'train': args.train,
    'evaluation': args.evaluation,
    'dnn': args.dnn,
    'bdt': args.bdt,
    'do_jetsyst': args.jetsyst,
    'dnn_bins': dnn_bins,
    'do_massscan': False,
    'mass': 125.0,
}

print(f"Start!")
print(f"Running options: {options}")
tstart = time.time() 

if load_unbinned_data:
    print(f"Will run modules: {modules}")
    for ptvar in all_pt_variations:
        postproc_args['syst_variations'] = [ptvar]
        print(f"Processing pt variation: {ptvar}")
        print(f"Getting unbinned data...")
        dfs, hist_dfs, edges = postprocess(postproc_args, not args.iterative)

        if args.train:
            print(f"Concatenating dataframes...")
            df = pd.concat(dfs)
            print("Starting DNN training...")
            classifier_train(df, postproc_args)
            print("DNN training complete!")
            sys.exit()

        if args.dnn_rebin:
            print("Rebinning DNN...")
            boundaries = dnn_rebin(dfs, postproc_args)
            print(args.year, args.label, boundaries)
            sys.exit()
    
        hist = {}
#        print(hist_dfs)
        for var, hists in hist_dfs.items():
            print(f"Concatenating histograms: {var}")
            hist[var] = pd.concat(hists, ignore_index=True)

        #print(f"Saving yields...")
        #save_yields(vars_to_plot['dimuon_mass'], hist, edges[myvar], postproc_args)
        if args.evaluation:
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
        plot(var, hist, edges[vname], postproc_args)
        for r in postproc_args['regions']:
            plot(var, hist, edges[vname], postproc_args, r)

elapsed = time.time() - tstart
print(f"Total time: {elapsed} s")
