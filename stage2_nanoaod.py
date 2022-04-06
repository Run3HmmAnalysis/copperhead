import glob
import tqdm
import argparse
import dask
from dask.distributed import Client
import dask.dataframe as dd

from nanoaod.postprocessor import load_dataframe

from nanoaod.postprocessor import training_features
from nanoaod.config.mva_bins import mva_bins
from nanoaod.config.variables import variables_lookup
from python.convert import to_histograms
from python.plotter import plotter

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
    "label": "2022mar28",
    "path": "/depot/cms/hmm/coffea/",
    "hist_path": "/depot/cms/hmm/coffea/histograms/",
    "plots_path": "./plots/2022mar28/",
    "models_path": "/depot/cms/hmm/trained_models/",
    "dnn_models": [],
    # "dnn_models": ["dnn_allyears_128_64_32"],
    # keep in mind - xgboost version for evaluation
    # should be exactly the same as the one used for training!
    # 'bdt_models': ['bdt_nest10000_weightCorrAndShuffle_2Aug'],
    "bdt_models": [],
    "do_massscan": False,
    "years": args.years,
    "syst_variations": ["nominal"],
    # 'syst_variations':['nominal', f'Absolute2016_up'],
    "channels": ["vbf"],
    "regions": ["h-peak", "h-sidebands"],
    "save_hists": True,
    "save_plots": True,
    "plot_ratio": True,
    "14TeV_label": False,
    "has_variations": True,
    "variables_lookup": variables_lookup,
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
    # "ewk_lljj_mll105_160_ptj0": "EWK",
    "ewk_lljj_mll105_160_py_dipole": "EWK",
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
# parameters["grouping"] = {"vbf_powheg_dipole": "VBF",}

parameters["plot_groups"] = {
    "stack": ["DY", "EWK", "TT+ST", "VV", "VVV"],
    "step": ["VBF", "ggH"],
    "errorbar": ["Data"],
}


if __name__ == "__main__":
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
    parameters["ncpus"] = len(client.scheduler_info()["workers"])
    print(f"Cluster created! #CPUs = {parameters['ncpus']}")

    datasets = parameters["grouping"].keys()

    # parameters["hist_vars"] = ["dimuon_mass", "jj_mass", "jj_dEta", "mu1_pt", "mu2_pt", "jet1_pt", "jet2_pt"]
    parameters["hist_vars"] = training_features + [
        "mu1_pt",
        "mu2_pt",
        "jet1_pt",
        "jet2_pt",
    ]
    parameters["hist_vars"] += ["score_" + m for m in parameters["dnn_models"]]
    parameters["hist_vars"] += ["score_" + m for m in parameters["bdt_models"]]

    # parameters['plot_vars'] = ['dimuon_mass']
    parameters["plot_vars"] = parameters["hist_vars"]
    parameters["datasets"] = datasets

    all_paths = {}
    for year in parameters["years"]:
        all_paths[year] = {}
        for dataset in datasets:
            paths = glob.glob(
                f"{parameters['path']}/"
                f"{year}_{parameters['label']}/"
                f"{dataset}/*.parquet"
            )
            all_paths[year][dataset] = paths

    if args.remake_hists:
        for year in parameters["years"]:
            print(f"Processing {year}")
            for dataset, path in tqdm.tqdm(all_paths[year].items()):
                if len(path) == 0:
                    continue
                df = load_dataframe(client, parameters, inputs=[path])
                if not isinstance(df, dd.DataFrame):
                    continue
                to_histograms(client, parameters, df=df)

    if args.plot:
        yields = plotter(client, parameters)
