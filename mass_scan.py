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
    '2016': [0, 0.188, 0.371, 0.548, 0.718, 0.883, 1.043, 1.196, 1.343, 1.485, 1.62, 1.75, 1.874, 2.3],
    '2017': [0, 0.23, 0.449, 0.657, 0.853, 1.038, 1.211, 1.373, 1.523, 1.662, 1.789, 1.905, 2.01, 2.3],
    '2018': [0, 0.22, 0.43, 0.63, 0.82, 0.999, 1.168, 1.327, 1.476, 1.614, 1.742, 1.86, 1.968, 2.3]
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
    'ggh_amcPS',
    'vbf_powhegPS',
    'vbf_powheg_herwig',
    'vbf_powheg_dipole'
]
samples_ = [
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
    'background': ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160_ptj0'],#, 'ttjets_dl'],
    'signal': ['ggh_amcPS','vbf_powhegPS', 'vbf_powheg_herwig'],
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

if (not args.jetsyst) or args.dnn_training or args.dnn_rebin:
    all_pt_variations = ['nominal']

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
    if not args.dnn_training: samples = ['vbf_powheg_dipole', 'ggh_amcPS']
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


postproc_args = {
    'modules': modules,
    'year': args.year,
    'label': args.label,
    'in_path': f'/depot/cms/hmm/coffea/{args.year}_{args.label}/',
    'syst_variations': all_pt_variations,
    'out_path': 'plots_new/',
    'samples':samples,
    'training_samples': training_samples,
    'channels': ['vbf','vbf_01j','vbf_2j'],
    'channel_groups': {'vbf':['vbf','vbf_01j','vbf_2j']},
    'regions': ['h-peak', 'h-sidebands'],
    'vars_to_plot': list(vars_to_plot.values()),
    'wgt_variations': True,
    'training': args.dnn_training,
    'do_jetsyst': args.jetsyst,
    'dnn_bins': dnn_bins,
    'do_massscan': True,
    'mass': 125.0
}

print(f"Start!")
print(f"Running options: {options}")
tstart = time.time() 

mass_points = [125.0, 125.1, 125.2, 125.3, 125.4, 125.5, 125.6, 125.7, 125.8, 125.9, 126.0]
#mass_points = [126.0]
for m in mass_points:
    postproc_args['mass'] = m
    print(f"Processing mass point: {m}")
    if load_unbinned_data:
        print(f"Will run modules: {modules}")
        for ptvar in all_pt_variations:
            postproc_args['syst_variations'] = [ptvar]
            print(f"Processing pt variation: {ptvar}")
            print(f"Getting unbinned data...")
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
    #        print(hist_dfs)
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
