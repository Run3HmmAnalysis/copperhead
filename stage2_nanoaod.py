import glob
import tqdm
import argparse
import dask
from dask.distributed import Client

from python.timer import Timer
from nanoaod.postprocessor import load_dataframe, to_histograms
from nanoaod.config.mva_bins import mva_bins
from plotting.plotter import plotter

__all__ = ["dask"]


parser = argparse.ArgumentParser()
parser.add_argument(
    "-sl",
    "--slurm",
    dest="slurm_port",
    default=None,
    action="store",
    help="Slurm cluster port (if not specified, will create a local cluster)",
)
parser.add_argument(
    "-s",
    "--sequential",
    dest="sequential",
    default=False,
    action="store_true",
    help="Sequential processing",
)
parser.add_argument(
    "-r",
    "--remake_hists",
    dest="remake_hists",
    default=False,
    action="store_true",
    help="Remake histograms",
)
parser.add_argument(
    "-p",
    "--plot",
    dest="plot",
    default=False,
    action="store_true",
    help="Produce plots",
)
parser.add_argument(
    "-y", "--years", nargs="+", help="Years to process", default=["2018"]
)

args = parser.parse_args()

use_local_cluster = args.slurm_port is None
ncpus_local = 40  # number of cores to use. Each one will start with 4GB

node_ip = "128.211.149.133"

# only if Slurm cluster is used:
slurm_cluster_ip = f"{node_ip}:{args.slurm_port}"

# if Local cluster is used:
dashboard_address = f"{node_ip}:34875"
if not use_local_cluster:
    dashboard_address = f"{node_ip}:8787"
# How to open the dashboard:
# Dashboard IP should be the same as the node
# from which the code is running. In my case it's hammer-c000.
# Port can be anything, you can use default :8787
# 1. Open remote desktop on Hammer (ThinLinc)
# 2. Launch terminal
# 3. Log it to the node from which you are running the processing,
# e.g. [ssh -Y hammer-c000]
# 4. Launch Firefox [firefox]
# 5. Type in Firefox the address: <dashboard_address>/status
# 6. Refresh few times once you started running plot_dask.py,
# the plots will appear

parameters = {
    "slurm_cluster_ip": slurm_cluster_ip,
    "ncpus": ncpus_local,
    "label": "sep26",
    "path": "/depot/cms/hmm/coffea/",
    "hist_path": "/depot/cms/hmm/coffea/histograms/",
    "plots_path": "./plots_test/",
    "models_path": "/depot/cms/hmm/trained_models/",
    "dnn_models": ["dnn_allyears_128_64_32"],
    # keep in mind - xgboost version for evaluation
    # should be exactly the same as the one used for training!
    # 'bdt_models': ['bdt_nest10000_weightCorrAndShuffle_2Aug'],
    "bdt_models": [],
    "do_massscan": False,
    "years": args.years,
    "syst_variations": ["nominal"],
    # 'syst_variations':['nominal', f'Absolute2016_up'],
    "channels": ["vbf", "vbf_01j", "vbf_2j"],
    # 'channels': ['ggh_01j', 'ggh_2j'],
    "regions": ["h-peak", "h-sidebands"],
    "save_hists": True,
    "save_plots": True,
    "plot_ratio": True,
    "14TeV_label": False,
    "has_variations": True,
}

parameters["mva_bins"] = mva_bins
parameters["grouping"] = {
    "data_A": "Data",
    "data_B": "Data",
    "data_C": "Data",
    "data_D": "Data",
    "data_E": "Data",
    "data_F": "Data",
    "data_G": "Data",
    "data_H": "Data",
    # 'dy_0j': 'DY',
    # 'dy_1j': 'DY',
    # 'dy_2j': 'DY',
    # 'dy_m105_160_amc': 'DY_nofilter',
    # 'dy_m105_160_vbf_amc': 'DY_filter',
    "dy_m105_160_amc": "DY",
    "dy_m105_160_vbf_amc": "DY",
    "ewk_lljj_mll105_160_ptj0": "EWK",
    # 'ewk_lljj_mll105_160_py_dipole': 'EWK_Pythia',
    "ttjets_dl": "TT+ST",
    "ttjets_sl": "TT+ST",
    "ttw": "TT+ST",
    "ttz": "TT+ST",
    "st_tw_top": "TT+ST",
    "st_tw_antitop": "TT+ST",
    "ww_2l2nu": "VV",
    "wz_2l2q": "VV",
    "wz_1l1nu2q": "VV",
    "wz_3lnu": "VV",
    "zz": "VV",
    "www": "VVV",
    "wwz": "VVV",
    "wzz": "VVV",
    "zzz": "VVV",
    "ggh_amcPS": "ggH",
    "vbf_powheg_dipole": "VBF",
}

grouping_alt = {
    "Data": [
        "data_A",
        "data_B",
        "data_C",
        "data_D",
        "data_E",
        "data_F",
        "data_G",
        "data_H",
    ],
    "DY": ["dy_m105_160_amc", "dy_m105_160_vbf_amc"],
    "EWK": ["ewk_lljj_mll105_160_ptj0"],
    "TT+ST": ["ttjets_dl", "ttjets_sl", "ttw", "ttz", "st_tw_top", "st_tw_antitop"],
    "VV": ["ww_2l2nu", "wz_2l2q", "wz_1l1nu2q", "wz_3lnu", "zz"],
    "VVV": ["www", "wwz", "wzz", "zzz"],
    "ggH": ["ggh_amcPS"],
    "VBF": ["vbf_powheg_dipole"],
}

parameters["plot_groups"] = {
    "stack": ["DY", "EWK", "TT+ST", "VV", "VVV"],
    "step": ["VBF", "ggH"],
    "data": ["Data"],
}


if __name__ == "__main__":
    timer = Timer(ordered=False)

    if use_local_cluster:
        print(
            f"Creating local cluster with {ncpus_local} workers."
            f" Dashboard address: {dashboard_address}"
        )
        client = Client(
            processes=True,
            dashboard_address=dashboard_address,
            n_workers=ncpus_local,
            threads_per_worker=1,
            memory_limit="4GB",
        )
    else:
        print(
            f"Creating Slurm cluster at {slurm_cluster_ip}."
            f" Dashboard address: {dashboard_address}"
        )
        client = Client(parameters["slurm_cluster_ip"])
    print("Cluster created!")

    datasets = [
        "ggh_amcPS",
        # 'vbf_powhegPS',
        # 'vbf_powheg_herwig',
        "vbf_powheg_dipole"
        # 'dy_m105_160_amc'
    ]
    datasets = parameters["grouping"].keys()

    parameters["hist_vars"] = ["dimuon_mass"]
    parameters["hist_vars"] += ["score_" + m for m in parameters["dnn_models"]]
    parameters["hist_vars"] += ["score_" + m for m in parameters["bdt_models"]]

    # parameters['plot_vars'] = ['dimuon_mass']
    parameters["plot_vars"] = parameters["hist_vars"]
    parameters["datasets"] = datasets

    how = {
        "Data": "grouped",
        "DY": "all",
        "EWK": "all",
        "TT+ST": "individual",
        "VV": "individual",
        "VVV": "individual",
        "ggH": "all",
        "VBF": "all",
    }
    paths_grouped = {}
    all_paths = []
    for y in parameters["years"]:
        paths_grouped[y] = {}
        paths_grouped[y]["all"] = []
        for group, ds in grouping_alt.items():
            for dataset in ds:
                if dataset not in datasets:
                    continue
                if how[group] == "all":
                    the_group = "all"
                elif how[group] == "grouped":
                    the_group = group
                elif how[group] == "individual":
                    the_group = dataset
                path = glob.glob(
                    f"{parameters['path']}/"
                    f"{y}_{parameters['label']}/"
                    f"{dataset}/*.parquet"
                )
                if the_group not in paths_grouped.keys():
                    paths_grouped[y][the_group] = []
                all_paths.append(path)
                paths_grouped[y][the_group].append(path)

    if args.remake_hists:
        if args.sequential:
            for path in tqdm.tqdm(all_paths):
                if len(path) == 0:
                    continue
                df = load_dataframe(client, parameters, inputs=[path], timer=timer)

        else:
            for year, groups in paths_grouped.items():
                print(f"Processing {year}")
                for group, g_paths in tqdm.tqdm(groups.items()):
                    if len(g_paths) == 0:
                        continue
                    df = load_dataframe(client, parameters, inputs=g_paths, timer=timer)
        to_histograms(client, parameters, df=df)

    if args.plot:
        plotter(client, parameters, timer=timer)
    timer.summary()
