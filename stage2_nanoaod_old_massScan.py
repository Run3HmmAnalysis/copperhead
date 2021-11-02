import time
import sys
import argparse
from nanoaod.postprocessor_old import postprocess, save_shapes
from nanoaod.postprocessor_old import make_datacards, prepare_root_files
from nanoaod.postprocessor_old import overlap_study_unbinned
from nanoaod.config.variables import Variable
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", dest="year", default="", action="store")
parser.add_argument("-l", "--label", dest="label", default="jun16", action="store")
parser.add_argument("-dnn", "--dnn", action="store_true")
parser.add_argument("-bdt", "--bdt", action="store_true")
parser.add_argument("-dc", "--datacards", action="store_true")
parser.add_argument("-js", "--jetsyst", action="store_true")
parser.add_argument("-i", "--iterative", action="store_true")
parser.add_argument("-o", "--overlap", action="store_true")

args = parser.parse_args()

fail = int(args.dnn) + int(args.bdt) + int(args.datacards) + int(args.overlap) == 0
if fail:
    print("Please specify option(s) to run:")
    print("-t --dnn_training")
    print("-dnn --dnn")
    print("-bdt --bdt")
    print("-dc --datacards")
    print("-o --overlap")
    sys.exit()

if (args.year == "") and not args.overlap:
    print("Year must be specified!")
    sys.exit()

mva_bins = {
    "dnn_allyears_128_64_32": {
        "2016": [
            0,
            0.16,
            0.374,
            0.569,
            0.75,
            0.918,
            1.079,
            1.23,
            1.373,
            1.514,
            1.651,
            1.784,
            1.923,
            2.8,
        ],
        "2017": [
            0,
            0.307,
            0.604,
            0.832,
            1.014,
            1.169,
            1.31,
            1.44,
            1.567,
            1.686,
            1.795,
            1.902,
            2.009,
            2.8,
        ],
        "2018": [
            0,
            0.07,
            0.432,
            0.71,
            0.926,
            1.114,
            1.28,
            1.428,
            1.564,
            1.686,
            1.798,
            1.9,
            2.0,
            2.8,
        ],
    },
    "bdt_nest10000_weightCorrAndShuffle_2Aug": {
        "2016": [
            0,
            0.282,
            0.57,
            0.802,
            0.999,
            1.171,
            1.328,
            1.479,
            1.624,
            1.775,
            1.93,
            2.097,
            2.288,
            5.0,
        ],
        "2017": [
            0,
            0.411,
            0.744,
            0.99,
            1.185,
            1.352,
            1.504,
            1.642,
            1.784,
            1.924,
            2.07,
            2.222,
            2.398,
            5.0,
        ],
        "2018": [
            0,
            0.129,
            0.621,
            0.948,
            1.189,
            1.388,
            1.558,
            1.717,
            1.866,
            2.01,
            2.152,
            2.294,
            2.451,
            5.0,
        ],
    },
}

dnn_models = ["dnn_allyears_128_64_32"]  # best DNN so far
# bdt_models = []
bdt_models = [
    # 'bdt_nest10000_weightCorrAndShuffle_2Aug',
]

vars_to_save = []
vars_to_plot = {}

models = []

if args.dnn or args.bdt:
    if args.dnn:
        models += dnn_models
    else:
        dnn_models = []
    if args.bdt:
        models += bdt_models
    else:
        bdt_models = []
else:
    models = dnn_models + bdt_models

for model in models:
    name = f"score_{model}"
    vars_to_plot.update({name: Variable(name, name, 50, 0, 5)})
    vars_to_save.append(vars_to_plot[name])

samples = [
    "data_A",
    "data_B",
    "data_C",
    "data_D",
    "data_E",
    "data_F",
    "data_G",
    "data_H",
    # 'ggh_amcPS',
    # 'vbf_powhegPS',
    # 'vbf_powheg_herwig',
    "vbf_powheg_dipole",
]

samples_ = [
    "data_A",
    "data_B",
    "data_C",
    "data_D",
    "data_E",
    "data_F",
    "data_G",
    "data_H",
    "dy_m105_160_amc",
    "dy_m105_160_vbf_amc",
    "ewk_lljj_mll105_160_ptj0",
    "ewk_lljj_mll105_160",
    "ewk_lljj_mll105_160_py",
    "ewk_lljj_mll105_160_py_dipole",
    "ttjets_dl",
    "ttjets_sl",
    "ttz",
    "ttw",
    "st_tw_top",
    "st_tw_antitop",
    "ww_2l2nu",
    "wz_2l2q",
    "wz_3lnu",
    "zz",
]

pt_variations = []
pt_variations += ["nominal"]
pt_variations += ["Absolute", f"Absolute{args.year}"]
pt_variations += ["BBEC1", f"BBEC1{args.year}"]
pt_variations += ["EC2", f"EC2{args.year}"]
pt_variations += ["FlavorQCD"]
pt_variations += ["HF", f"HF{args.year}"]
pt_variations += ["RelativeBal", f"RelativeSample{args.year}"]
pt_variations += ["jer1", "jer2", "jer3"]
pt_variations += ["jer4", "jer5", "jer6"]

all_pt_variations = []
for ptvar in pt_variations:
    if ptvar == "nominal":
        all_pt_variations += ["nominal"]
    else:
        all_pt_variations += [f"{ptvar}_up"]
        all_pt_variations += [f"{ptvar}_down"]

if (not args.jetsyst) or args.overlap:
    all_pt_variations = ["nominal"]


# keeping correct order just for debug output
def add_modules(modules, new_modules):
    for m in new_modules:
        if m not in modules:
            modules.append(m)
    return modules


load_unbinned_data = True
modules = []
options = []
if args.dnn:
    if len(dnn_models) == 0:
        print("List of DNN models is empty!")
        sys.exit()
    modules = add_modules(modules, ["to_pandas", "evaluation", "get_hists"])
    options += ["evaluation"]
if args.bdt:
    if len(bdt_models) == 0:
        print("List of BDT models is empty!")
        sys.exit()
    modules = add_modules(modules, ["to_pandas", "evaluation", "get_hists"])
    options += ["evaluation"]
if args.datacards:
    options += ["datacards"]
    if not (args.dnn or args.bdt):
        load_unbinned_data = False
if args.overlap:
    modules = add_modules(modules, ["to_pandas", "evaluation"])
    options += ["dnn_overlap"]

postproc_args = {
    "modules": modules,
    "year": args.year,
    "label": args.label,
    "in_path": "/depot/cms/hmm/coffea/",
    "dnn_models": dnn_models,
    "bdt_models": bdt_models,
    "syst_variations": all_pt_variations,
    "out_path": "plots_new/",
    "samples": samples,
    "training_samples": {},
    "channels": ["vbf", "vbf_01j", "vbf_2j"],
    "channel_groups": {"vbf": ["vbf", "vbf_01j", "vbf_2j"]},
    "regions": ["h-peak", "h-sidebands"],
    "vars_to_plot": list(vars_to_plot.values()),
    "wgt_variations": True,
    "training": False,
    "do_jetsyst": args.jetsyst,
    "mva_bins": mva_bins,
    "do_massscan": True,
    "mass": 125.0,
}

print("Start!")
print(f"Running options: {options}")
tstart = time.time()

mass_points = [
    120.0,
    120.5,
    121.0,
    121.5,
    122.0,
    122.5,
    123.0,
    123.5,
    124.0,
    124.5,
    125.0,
    125.1,
    125.2,
    125.3,
    125.38,
    125.4,
    125.5,
    125.6,
    125.7,
    125.8,
    125.9,
    126.0,
    126.5,
    127.0,
    127.5,
    128.0,
    128.5,
    129.0,
    129.5,
    130.0,
]
mass_points = [125.0]  # , 125.38, 125.5, 126.0]

num_options = len(mass_points) * len(all_pt_variations)
iopt = 0
for m in mass_points:
    hist = {}
    postproc_args["mass"] = m
    print(f"Running mass point: {m}")
    if load_unbinned_data:
        print(f"Will run modules: {modules}")
        for ptvar in all_pt_variations:
            postproc_args["syst_variations"] = [ptvar]
            print(f"Running option: {args.year} m={m} {ptvar}")
            iopt += 1
            print(f"This is option #{iopt} out of {num_options}" "for {args.year}")
            print("Getting unbinned data...")
            dfs, hist_dfs, edges = postprocess(postproc_args, not args.iterative)

            if args.overlap:
                print("Studying overlap with Pisa...")
                df = pd.concat(dfs)
                for model in models:
                    # overlap_study(df, postproc_args, model)
                    overlap_study_unbinned(df, postproc_args, model)
                continue

            for var, hists in hist_dfs.items():
                print(f"Concatenating histograms: {var}")
                if var not in hist.keys():
                    hist[var] = pd.concat(hists, ignore_index=True)
                else:
                    hist[var] = pd.concat(hists + [hist[var]], ignore_index=True)

            if args.dnn or args.bdt:
                for model in models:
                    print(f"Saving shapes: {model}")
                    save_shapes(hist, model, postproc_args, mva_bins)

    if args.datacards:
        for myvar in vars_to_save:
            print(f"Preparing ROOT files with shapes ({myvar.name})...")
            prepare_root_files(myvar, postproc_args, shift_signal=False)
            prepare_root_files(myvar, postproc_args, shift_signal=True)
            print("Writing datacards...")
            make_datacards(myvar, postproc_args, shift_signal=False)
            make_datacards(myvar, postproc_args, shift_signal=True)

elapsed = time.time() - tstart
print(f"Total time: {elapsed} s")
