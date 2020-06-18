import os
import uproot, coffea
import numpy as np
import argparse
from config.variables import variables
import matplotlib.pyplot as plt
import boost_histogram as bh
import mplhep as hep
from python.postprocessing import var_map_pisa

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", dest="year", default='2018', action='store')
parser.add_argument("-l", "--label", dest="label", default="jun16", action='store')
args = parser.parse_args()

try:
    os.mkdir(f"compare_with_pisa/plots_{args.year}_{args.label}_missing")
except:
    pass

with uproot.open('/depot/cms/hmm/coffea/pisa-jun12/data2018Snapshot.root') as f:
    tree = f['Events']
    pisa_events = tree['event'].array()
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

    for var in variables:
        if var.name not in var_map_pisa.keys(): continue
        hist = bh.Histogram(bh.axis.Regular(var.nbins, var.xmin, var.xmax))
        hist.fill(tree[var_map_pisa[var.name]].array()[filter_missing])
        data_hist = hist.to_numpy()[0]
        edges = hist.to_numpy()[1]

        fig = plt.figure()
        plt.rcParams.update({'font.size': 22})
        data_opts_pisa = {'color': 'red', 'marker': '.', 'markersize':15}
        fig.clf()
        fig.set_size_inches(12,12)

        plt1 = fig.add_subplot(111)
        ax = hep.histplot(data_hist, edges, ax=plt1,  histtype='errorbar', yerr=np.sqrt(data_hist), **data_opts_pisa)

        plt1.set_yscale('log')
        plt1.set_ylim(0.01, 1e9)
        plt1.set_xlim(edges[0],edges[-1])
        plt1.set_xlabel(var.name)

        out_name = f"compare_with_pisa/plots_{args.year}_{args.label}_missing/{var.name}_h-peak.png"
        print(f"Saving {out_name}")
        fig.savefig(out_name)
