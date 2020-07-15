import time
import os,sys,glob
import argparse
from python.postprocessing import postprocess, plot, save_yields, load_yields, save_shapes, make_datacards, rebin, dnn_training, prepare_root_files, overlap_study
from config.variables import variables
from config.datasets import datasets
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", dest="year", default='', action='store')
parser.add_argument("-l", "--label", dest="label", default="jun16", action='store')
parser.add_argument("-t", "--dnn_training", action='store_true')
parser.add_argument("-r", "--rebin", action='store_true')
parser.add_argument("-dnn", "--dnn_evaluation", action='store_true')
parser.add_argument("-bdt", "--bdt_evaluation", action='store_true')
parser.add_argument("-dc", "--datacards", action='store_true')
parser.add_argument("-p", "--plot", action='store_true')
parser.add_argument("-o", "--overlap", action='store_true')
parser.add_argument("-js", "--jetsyst", action='store_true')
parser.add_argument("-i", "--iterative", action='store_true')
args = parser.parse_args()

if int(args.dnn_training)+int(args.rebin)+int(args.dnn_evaluation)+int(args.bdt_evaluation)+\
int(args.datacards)+int(args.plot)+int(args.overlap)==0:
    print("Please specify option(s) to run:")
    print("-t --dnn_training")
    print("-r --rebin")
    print("-dnn --dnn_evaluation")
    print("-bdt --bdt_evaluation")
    print("-dc --datacards")
    print("-p --plot")
    print("-o --overlap")
    sys.exit()

if (args.year=='') and not args.dnn_training:
    print("Year must be specified! Merging data from different years is only allowed for DNN training.")
    sys.exit()

if args.dnn_training and (args.dnn_evaluation or args.bdt_evaluation or args.datacards or args.plot or args.overlap):
    print("Can't combine 'training' option with 'evaluation','datacards' or 'plot'!")
    sys.exit()

if args.rebin and (args.dnn_evaluation or args.bdt_evaluation or args.datacards or args.plot or args.overlap):
    print("Can't combine 'rebin' option with 'evaluation','datacards' or 'plot'!")
    sys.exit()

if (args.dnn_evaluation or args.bdt_evaluation) and (args.dnn_training or args.rebin or args.overlap):
    print("Can't combine 'evaluation' option with 'training' or 'rebin'!")
    sys.exit()

if args.datacards and (args.dnn_training or args.rebin or args.plot or args.overlap):
    print("Can't combine 'datacards' option with 'training', 'rebin' or 'plot'!")
    sys.exit()

if args.plot and (args.dnn_training or args.rebin or args.datacards or args.overlap):
    print("Can't combine 'datacards' option with 'training', 'rebin' or 'datacards'!")
    sys.exit()

if args.overlap and (args.dnn_evaluation or args.bdt_evaluation or args.dnn_training or args.rebin or args.datacards or args.plot):
    print("Can't combine overlap study with other options!")
    sys.exit()


    
#to_plot = ['dimuon_mass', 'dimuon_pt', 'mu1_pt', 'jet1_pt', 'jet1_eta', 'jet2_pt', 'jet2_eta', 'dnn_score']
to_plot = ['dimuon_mass', 'dnn_score', 'bdt_score']

vars_to_plot = {v.name:v for v in variables if v.name in to_plot}
#vars_to_plot = {v.name:v for v in variables}

dnn_score = [v for v in variables if (v.name == 'dnn_score')][0] # variable for templates and datacards
bdt_score = [v for v in variables if (v.name == 'bdt_score')][0] # variable for templates and datacards
vars_to_save = [dnn_score, bdt_score]

mva_bins = {
    'dnn_nominal':{
        '2016': [0, 0.158, 0.379, 0.582, 0.77, 0.945, 1.111, 1.265, 1.413, 1.553, 1.68, 1.796, 1.871, 2.3],
        '2017': [],
        '2018': []
        },
    'dnn_nominal_allyears': {
        '2016': [0, 0.172, 0.404, 0.608, 0.795, 0.972, 1.137, 1.293, 1.446, 1.597, 1.743, 1.881, 1.987, 2.3],
        '2017': [0, 0.335, 0.641, 0.879, 1.072, 1.238, 1.388, 1.527, 1.663, 1.782, 1.88,  1.957, 2.017, 2.3],
        '2018': [0, 0.075, 0.462, 0.75, 0.976, 1.171, 1.35, 1.513, 1.664, 1.794, 1.896, 1.969, 2.021, 2.3]
    },
    'bdt_jul15_earlystop50':{
        '2016': [0, 0.54, 0.544, 0.547, 0.549, 0.551, 0.553, 0.555, 0.557, 0.558, 0.56, 0.562, 0.564, 2.8],
        '2017': [],
        '2018': [],
    }
}

dnn_model = ''
bdt_model = ''

dnn_model = 'dnn_nominal_allyears'
#bdt_model = 'bdt_jul15_earlystop50'

samples_ = [
    'ttjets_dl',
    'ttjets_sl',
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
    'background': ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160_ptj0'],
    'signal': ['vbf_powhegPS','vbf_powheg_herwig', 'ggh_amcPS'],
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

if (not args.jetsyst) or args.dnn_training or args.rebin or args.overlap:
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
if args.rebin:
    modules = add_modules(modules,['to_pandas', 'evaluation'])
    if not args.dnn_training: samples = ['vbf_powheg_dipole', 'ggh_amcPS']
    options += ['rebin']
if args.overlap:
    modules = add_modules(modules,['to_pandas', 'evaluation'])
    samples = ['vbf_powhegPS','vbf_powheg']
    options += ['dnn_overlap']
if args.dnn_evaluation or args.bdt_evaluation:
    modules = add_modules(modules,['to_pandas', 'evaluation', 'get_hists'])
    options += ['evaluation']
    if not args.plot:
        vars_to_plot = {}
        if args.dnn_evaluation:
            vars_to_plot.update({v.name:v for v in variables if v.name in ['dnn_score']})
        if args.bdt_evaluation:
            vars_to_plot.update({v.name:v for v in variables if v.name in ['bdt_score']})
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
    'in_path': f'/depot/cms/hmm/coffea/',
    'dnn': (args.dnn_evaluation or args.rebin or (args.overlap and dnn_model!='')),
    'dnn_model': dnn_model,
    'evaluate_allyears_dnn': ('allyears' in dnn_model),
    'bdt': (args.bdt_evaluation or args.rebin or (args.overlap and bdt_model!='')),
    'bdt_model': bdt_model,
    'evaluate_allyears_bdt': ('allyears' in bdt_model),
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
    'mva_bins': mva_bins,
    'do_massscan': False,
    'mass': 125.0,
}

print(f"Start!")
print(f"Running options: {options}")
tstart = time.time() 

hist = {}

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

        if args.rebin:
            print("Rebinning DNN/BDT...")
            df = pd.concat(dfs)
            boundaries = rebin(df, postproc_args, 'dnn_score')
            print(args.year, args.label, dnn_model, boundaries)
            boundaries = rebin(df, postproc_args, 'bdt_score')
            print(args.year, args.label, bdt_model, boundaries)            
            sys.exit()

        if args.overlap:
            print("Studying overlap with Pisa...")
            df = pd.concat(dfs)
            overlap_study(df, postproc_args, dnn_model, 'dnn_score')
            overlap_study(df, postproc_args, bdt_model, 'bdt_score')
            sys.exit()

#        print(hist_dfs)
        for var, hists in hist_dfs.items():
            print(f"Concatenating histograms: {var}")
            if var not in hist.keys():
                hist[var] = pd.concat(hists, ignore_index=True)
            else:
                hist[var] = pd.concat(hists+[hist[var]], ignore_index=True)

        if args.dnn_evaluation and not args.plot:
            print(f"Saving shapes: {dnn_score.name}")
            save_shapes(dnn_score, hist, postproc_args)
        if args.bdt_evaluation and not args.plot:
            print(f"Saving shapes: {bdt_score.name}")
            save_shapes(bdt_score, hist, postproc_args)


            
if args.datacards:
    for myvar in vars_to_save:
        print(f"Preparing ROOT files with shapes ({myvar.name})...")    
        prepare_root_files(myvar, postproc_args)
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
