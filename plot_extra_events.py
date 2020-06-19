import os,glob, sys
import argparse
from python.postprocessing import postprocess, plot, save_shapes, make_datacards, dnn_rebin, var_map_pisa
from config.variables import variables
from config.datasets import datasets
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", dest="year", default='2018', action='store')
parser.add_argument("-l", "--label", dest="label", default="jun16", action='store')
args = parser.parse_args()

to_plot = list(var_map_pisa.keys())
vars_to_plot = {v.name:v for v in variables if v.name in to_plot}
samples = [
    'data_A',
    'data_B','data_C', 'data_D',
#    'dy_m105_160_amc',
#    'dy_m105_160_vbf_amc',
]

modules =  ['to_pandas',  'get_hists']        


postproc_args = {
    'modules': modules,
    'year': args.year,
    'label': args.label,
    'in_path': f'/depot/cms/hmm/coffea/{args.year}_{args.label}/',
    'syst_variations': ['nominal'],
    'out_path': 'compare_with_pisa/',
    'samples':samples,
    'channels': ['vbf','vbf_01j','vbf_2j'],
    'channel_groups': {'vbf':['vbf','vbf_01j','vbf_2j']},
    'regions': ['h-peak'],
    'vars_to_plot': list(vars_to_plot.values()),
    'wgt_variations': False,
    'do_jetsyst': False,
    'training': False,
    'dnn_bins': [],
}

import uproot, coffea
import numpy as np

with uproot.open('/depot/cms/hmm/coffea/pisa-jun12/data2018Snapshot.root') as f:
    pisa_events = f['Events']['event'].array()

purdue_events = []
for i in ['A','B','C','D']:
    fi = coffea.util.load(f'/depot/cms/hmm/coffea/{args.year}_{args.label}/nominal/data_{i}.coffea')
    purdue_events.extend(fi['event_vbf_h-peak'].value)
purdue_events = np.array(purdue_events)

filter_extra = ~np.isin(purdue_events,pisa_events)
extra_events = purdue_events[filter_extra]
print("Extra events: ", len(extra_events))

filter_missing = ~np.isin(pisa_events,purdue_events)
missing_events = pisa_events[filter_missing]
print("Missing events: ", len(missing_events))

postproc_args.update({'extra_events':extra_events})
postproc_args.update({'plot_extra': True})

dfs, hist_dfs, edges = postprocess(postproc_args)

hist = {}
for var, hists in hist_dfs.items():
    hist[var] = pd.concat(hists, ignore_index=True)

for vname, var in vars_to_plot.items():
    for r in postproc_args['regions']:
        plot(var, hist, edges[vname], postproc_args, r, compare_with_pisa=False)

