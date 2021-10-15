import time
import sys
import argparse
from nanoaod.postprocessor_old import postprocess, plot
from nanoaod.postprocessor_old import save_shapes, make_datacards
from nanoaod.postprocessor_old import rebin, dnn_training
from nanoaod.postprocessor_old import prepare_root_files
from nanoaod.postprocessor_old import overlap_study_unbinned, plot_rocs
from config.variables import variables, Variable
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", dest="year", default="", action="store")
parser.add_argument("-l", "--label", dest="label", default="jun16", action="store")
parser.add_argument("-t", "--dnn_training", action="store_true")
parser.add_argument("-r", "--rebin", action="store_true")
parser.add_argument("-dnn", "--dnn", action="store_true")
parser.add_argument("-bdt", "--bdt", action="store_true")
parser.add_argument("-dc", "--datacards", action="store_true")
parser.add_argument("-p", "--plot", action="store_true")
parser.add_argument("-roc", "--roc", action="store_true")
parser.add_argument("-o", "--overlap", action="store_true")
parser.add_argument("-js", "--jetsyst", action="store_true")
parser.add_argument("-i", "--iterative", action="store_true")
args = parser.parse_args()

if (
    int(args.dnn_training)
    + int(args.rebin)
    + int(args.dnn)
    + int(args.bdt)
    + int(args.datacards)
    + int(args.plot)
    + int(args.overlap)
    + int(args.roc)
    == 0
):
    print("Please specify option(s) to run:")
    print("-t --dnn_training")
    print("-r --rebin")
    print("-dnn --dnn")
    print("-bdt --bdt")
    print("-dc --datacards")
    print("-p --plot")
    print("-o --overlap")
    print("-roc --roc")
    sys.exit()

if (args.year == "") and not args.dnn_training and not args.roc and not args.overlap:
    print(
        "Year must be specified!"
        " Merging data from different years is only "
        "allowed for DNN training."
    )
    sys.exit()

if args.dnn_training and (
    args.dnn or args.bdt or args.datacards or args.plot or args.overlap
):
    print("Can't combine 'training' option with" " 'evaluation','datacards' or 'plot'!")
    sys.exit()

if args.rebin and (args.datacards or args.plot or args.overlap):
    print("Can't combine 'rebin' option with 'datacards' or 'plot'!")
    sys.exit()

if (args.dnn or args.bdt) and (args.dnn_training):
    print("Can't combine 'evaluation' option with" " 'training' or 'overlap'!")
    sys.exit()

if args.datacards and (args.dnn_training or args.rebin or args.plot or args.overlap):
    print("Can't combine 'datacards' option with 'training'," " 'rebin' or 'plot'!")
    sys.exit()

if args.plot and (args.dnn_training or args.rebin or args.datacards or args.overlap):
    print("Can't combine 'datacards' option with 'training'," "'rebin' or 'datacards'!")
    sys.exit()

if args.overlap and (
    args.dnn
    or args.bdt
    or args.dnn_training
    or args.rebin
    or args.datacards
    or args.plot
    or args.roc
):
    print("Can't combine overlap study with other options!")
    sys.exit()

if args.roc and (
    args.dnn_training or args.rebin or args.datacards or args.plot or args.overlap
):
    print(
        "Can't combine ROC plotting with other options"
        "(will add support of that later)!"
    )
    sys.exit()

mva_bins = {
    "dnn_nominal": {
        "2016": [
            0,
            0.158,
            0.379,
            0.582,
            0.77,
            0.945,
            1.111,
            1.265,
            1.413,
            1.553,
            1.68,
            1.796,
            1.871,
            2.3,
        ],
        "2017": [],
        "2018": [],
    },
    "dnn_nominal_allyears": {
        "2016": [
            0,
            0.172,
            0.404,
            0.608,
            0.795,
            0.972,
            1.137,
            1.293,
            1.446,
            1.597,
            1.743,
            1.881,
            1.987,
            2.3,
        ],
        "2017": [
            0,
            0.335,
            0.641,
            0.879,
            1.072,
            1.238,
            1.388,
            1.527,
            1.663,
            1.782,
            1.88,
            1.957,
            2.017,
            2.3,
        ],
        "2018": [
            0,
            0.075,
            0.462,
            0.75,
            0.976,
            1.171,
            1.35,
            1.513,
            1.664,
            1.794,
            1.896,
            1.969,
            2.021,
            2.3,
        ],
    },
    "dnn_nominal_allyears500": {
        "2016": [],
        "2017": [
            0,
            0.324,
            0.627,
            0.86,
            1.047,
            1.206,
            1.352,
            1.489,
            1.628,
            1.758,
            1.868,
            1.956,
            2.039,
            2.3,
        ],
        "2018": [],
    },
    "dnn_nominal_allyears_4layers": {
        "2016": [
            0,
            0.168,
            0.391,
            0.588,
            0.765,
            0.929,
            1.079,
            1.224,
            1.372,
            1.537,
            1.722,
            1.884,
            1.962,
            2.3,
        ],
        "2017": [
            0,
            0.319,
            0.615,
            0.841,
            1.018,
            1.17,
            1.306,
            1.438,
            1.588,
            1.748,
            1.873,
            1.937,
            1.974,
            2.3,
        ],
        "2018": [
            0,
            0.069,
            0.446,
            0.724,
            0.934,
            1.112,
            1.27,
            1.421,
            1.581,
            1.748,
            1.88,
            1.942,
            1.976,
            2.3,
        ],
    },
    "dnn_allyears_64_128_64": {
        "2016": [],
        "2017": [
            0,
            0.313,
            0.608,
            0.836,
            1.021,
            1.181,
            1.327,
            1.46,
            1.589,
            1.705,
            1.807,
            1.9,
            1.996,
            2.3,
        ],
        "2018": [],
    },
    "dnn_allyears_64_128_128_64": {
        "2016": [],
        "2017": [
            0,
            0.293,
            0.592,
            0.83,
            1.019,
            1.18,
            1.328,
            1.466,
            1.609,
            1.747,
            1.864,
            1.954,
            2.009,
            2.3,
        ],
        "2018": [],
    },
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
    "dnn_allyears_128_64_32_test": {
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
    "bdt_jul15_earlystop50": {
        "2016": [
            0,
            0.284,
            0.575,
            0.804,
            1.0,
            1.17,
            1.328,
            1.477,
            1.623,
            1.774,
            1.93,
            2.097,
            2.29,
            2.8,
        ],
        "2017": [
            0,
            0.414,
            0.747,
            0.991,
            1.182,
            1.349,
            1.499,
            1.637,
            1.777,
            1.917,
            2.062,
            2.216,
            2.392,
            2.8,
        ],
        "2018": [
            0,
            0.129,
            0.627,
            0.953,
            1.195,
            1.393,
            1.565,
            1.725,
            1.875,
            2.02,
            2.164,
            2.308,
            2.471,
            2.8,
        ],
    },
    "bdt_nest10000_bestmodel_31July": {
        "2016": [
            0,
            0.282,
            0.569,
            0.797,
            0.991,
            1.162,
            1.318,
            1.466,
            1.611,
            1.759,
            1.914,
            2.078,
            2.267,
            5.0,
        ],
        "2017": [
            0,
            0.41,
            0.742,
            0.989,
            1.182,
            1.35,
            1.501,
            1.64,
            1.781,
            1.921,
            2.066,
            2.219,
            2.395,
            5.0,
        ],
        "2018": [
            0,
            0.129,
            0.622,
            0.948,
            1.191,
            1.388,
            1.56,
            1.72,
            1.869,
            2.013,
            2.156,
            2.298,
            2.457,
            5.0,
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
    "bdt_nest10000_allyears_multiclass_clsweightAndShuffle_6Aug": {
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
dnn_model_to_train = "dnn_allyears_200_100_50"
dnn_models = [
    # 'dnn_nominal_allyears',
    # 'dnn_nominal_allyears500',
    # 'dnn_nominal_allyears_4layers',
    # 'dnn_allyears_64_128_64',
    # 'dnn_allyears_64_128_128_64',
    # 'dnn_allyears_32_64_32',
    # 'dnn_allyears_32_64_64_32',
    # 'dnn_allyears_32_64_32_16',
    # 'dnn_allyears_32_64_32_16_8',
    # 'dnn_allyears_64_32_16',
    # 'dnn_allyears_64_32_16_8',
    # 'dnn_allyears_32_16_8',
    # 'dnn_allyears_100_50_25',
    # 'dnn_allyears_64_16_4',
    # 'dnn_allyears_64_48_32',
    "dnn_allyears_128_64_32",  # best DNN so far
    # 'dnn_allyears_128_64_32_test',
    # 'dnn_allyears_200_100_50',
    # 'dnn_nominal_allyears_4layers',
    # 'dnn_nominal_allyears_5layers',
    # 'dnn_nominal_allyears_6layers',
    # 'dnn_nominal_allyears_7layers',
    # 'dnn_nominal_allyears_8layers',
    # 'dnn_nominal_allyears_9layers',
    # 'dnn_nominal_allyears_10layers'
]

# bdt_models = []
bdt_models = [
    # 'bdt_jul15_earlystop50',
    # 'bdt_nest10000_bestmodel_31July',
    # 'bdt_nest10000_weightCorrAndShuffle_2Aug',
    # 'bdt_nest10000_allyears_multiclass_clsweightAndShuffle_6Aug',
]

to_plot_ = [
    "dimuon_mass",
    "dimuon_mass_res",
    "dimuon_pt",
    "dimuon_eta",
    "dimuon_phi",
    "dimuon_cos_theta_cs",
    "dimuon_phi_cs",
    "mu1_pt",
    "mu1_eta",
    "mu1_phi",
    "mu2_pt",
    "mu2_eta",
    "mu2_phi",
    "jet1_pt",
    "jet1_eta",
    "jet1_phi",
    "jet1_qgl",
    "jet2_pt",
    "jet2_eta",
    "jet2_phi",
    "jet2_qgl",
    "jj_mass",
    "jj_deta",
    "nsoftjets5",
    "htsoft2",
]
to_plot = ["dimuon_mass"]
# to_plot = ['rpt','jet1_eta','jet2_eta']
vars_to_save = []

if args.plot:
    vars_to_plot = {v.name: v for v in variables if v.name in to_plot}
    # vars_to_plot = {v.name:v for v in variables}
else:
    vars_to_plot = {}

models = []

if args.overlap:
    models = dnn_models + bdt_models

if args.dnn:
    models += dnn_models
elif (not args.rebin) and (not args.overlap):
    dnn_models = []
if args.bdt:
    models += bdt_models
elif (not args.rebin) and (not args.overlap):
    bdt_models = []

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
    "vbf_powheg_dipole"
    # 'dy_m105_160_amc',
    # 'dy_m105_160_vbf_amc',
    # 'ewk_lljj_mll105_160_ptj0',
    # 'vbf_powhegPS',
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
    "ggh_amcPS",
    "vbf_powhegPS",
    "vbf_powheg_herwig",
    "vbf_powheg_dipole",
]

samples_for_roc = [
    "dy_m105_160_amc",
    "dy_m105_160_vbf_amc",
    "ewk_lljj_mll105_160_ptj0",
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
    "ggh_amcPS",
    "vbf_powheg_dipole",
]

training_samples = {
    "background": [
        "dy_m105_160_amc",
        "dy_m105_160_vbf_amc",
        "ewk_lljj_mll105_160_ptj0",
    ],
    "signal": ["vbf_powhegPS", "vbf_powheg_herwig", "ggh_amcPS"],
}

all_training_samples = []
for k, v in training_samples.items():
    all_training_samples.extend(v)

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

force_nominal = (
    (not args.jetsyst) or args.dnn_training or args.rebin or args.overlap or args.roc
)
if force_nominal:
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
if args.dnn_training:
    modules = add_modules(modules, ["to_pandas"])
    samples = all_training_samples
    options += ["dnn_training"]
if args.rebin:
    modules = add_modules(modules, ["to_pandas", "evaluation"])
    if not args.dnn_training:
        samples = ["vbf_powheg_dipole", "ggh_amcPS"]
    options += ["rebin"]
if args.overlap:
    modules = add_modules(modules, ["to_pandas", "evaluation"])
    # samples = ['vbf_powhegPS', 'vbf_powheg_dipole']
    options += ["dnn_overlap"]
if args.roc:
    modules = add_modules(modules, ["to_pandas", "evaluation"])
    samples = samples_for_roc
    options += ["dnn_roc"]
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
if args.plot:
    modules = add_modules(modules, ["to_pandas", "get_hists"])
    options += ["plot"]

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
    "training_samples": training_samples,
    "channels": ["vbf", "vbf_01j", "vbf_2j"],
    "channel_groups": {"vbf": ["vbf", "vbf_01j", "vbf_2j"]},
    "regions": ["h-peak", "h-sidebands"],
    "vars_to_plot": list(vars_to_plot.values()),
    "wgt_variations": True,
    "training": args.dnn_training,
    "do_jetsyst": args.jetsyst,
    "mva_bins": mva_bins,
    "do_massscan": False,
    "mass": 125.0,
}

print("Start!")
print(f"Running options: {options}")
tstart = time.time()

hist = {}

if load_unbinned_data:
    print(f"Will run modules: {modules}")
    for ptvar in all_pt_variations:
        postproc_args["syst_variations"] = [ptvar]
        print(f"Processing pt variation: {ptvar}")
        print("Getting unbinned data...")
        dfs, hist_dfs, edges = postprocess(postproc_args, not args.iterative)

        if args.dnn_training:
            print("Concatenating dataframes...")
            df = pd.concat(dfs)
            print("Starting DNN training...")
            dnn_training(df, postproc_args, dnn_model_to_train)
            print("DNN training complete!")
            sys.exit()

        if args.rebin:
            df = pd.concat(dfs)
            for model in dnn_models + bdt_models:
                boundaries = rebin(df, model, postproc_args)
                print(model, boundaries)
            sys.exit()

        if args.overlap:
            print("Studying overlap with Pisa...")
            df = pd.concat(dfs)
            for model in models:
                # overlap_study(df, postproc_args, model)
                overlap_study_unbinned(df, postproc_args, model)
            sys.exit()

        if args.roc:
            print("Plotting ROC curves...")
            df = pd.concat(dfs)
            plot_rocs(df, postproc_args)
            sys.exit()

        for var, hists in hist_dfs.items():
            print(f"Concatenating histograms: {var}")
            if var not in hist.keys():
                hist[var] = pd.concat(hists, ignore_index=True)
            else:
                hist[var] = pd.concat(hists + [hist[var]], ignore_index=True)

        if (args.dnn or args.bdt) and not args.plot:
            for model in models:
                print(f"Saving shapes: {model}")
                save_shapes(hist, model, postproc_args, mva_bins)

if args.datacards:
    for myvar in vars_to_save:
        print(f"Preparing ROOT files with shapes ({myvar.name})...")
        prepare_root_files(myvar, postproc_args)
        print("Writing datacards...")
        make_datacards(myvar, postproc_args)

if args.plot:
    for vname, var in vars_to_plot.items():
        print(f"Plotting: {vname}")
        plot(var, hist, edges[vname], postproc_args)
        for r in postproc_args["regions"]:
            plot(var, hist, edges[vname], postproc_args, r)

elapsed = time.time() - tstart
print(f"Total time: {elapsed} s")
