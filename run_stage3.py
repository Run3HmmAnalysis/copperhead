import argparse
import dask
from dask.distributed import Client

from config.variables import variables_lookup
from stage3.plotter import plotter

__all__ = ["dask"]


parser = argparse.ArgumentParser()
parser.add_argument(
    "-y", "--years", nargs="+", help="Years to process", default=["2018"]
)
parser.add_argument(
    "-sl",
    "--slurm",
    dest="slurm_port",
    default=None,
    action="store",
    help="Slurm cluster port (if not specified, will create a local cluster)",
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

parameters = {
    "slurm_cluster_ip": slurm_cluster_ip,
    "label": "2022apr6",
    "hist_path": "/depot/cms/hmm/coffea/histograms/",
    "plots_path": "./plots/2022apr10/",
    "dnn_models": [],
    "bdt_models": [],
    "years": args.years,
    "channels": ["vbf"],
    "regions": ["h-peak", "h-sidebands"],
    "save_plots": True,
    "plot_ratio": True,
    "variables_lookup": variables_lookup,
}

parameters["grouping"] = {
    "data_A": "Data",
    "data_B": "Data",
    "data_C": "Data",
    "data_D": "Data",
    "data_E": "Data",
    "data_F": "Data",
    "data_G": "Data",
    "data_H": "Data",
    "dy_m105_160_amc": "DY",
    "dy_m105_160_vbf_amc": "DY",
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

    parameters["plot_vars"] = ["dimuon_mass"]
    parameters["plot_vars"] += ["score_" + m for m in parameters["dnn_models"]]
    parameters["plot_vars"] += ["score_" + m for m in parameters["bdt_models"]]

    parameters["datasets"] = parameters["grouping"].keys()

    yields = plotter(client, parameters)
