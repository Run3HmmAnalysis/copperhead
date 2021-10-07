import glob
import tqdm
import dask
from dask.distributed import Client

from python.timer import Timer
from python.postprocessor import workflow, plotter, grouping_alt
from python.postprocessor import grouping
from mva_bins import mva_bins

__all__ = ["dask"]

use_local_cluster = True
remake_hists = True
grouped = True
# is False, will use Slurm cluster (requires manual setup of the cluster)

ncpus_local = 40  # number of cores to use. Each one will start with 4GB

# only if Slurm cluster is used:
slurm_cluster_ip = "128.211.149.133:45579"  # '128.211.149.133:34003'


# if Local cluster is used:
dashboard_address = "128.211.149.133:34875"
if not use_local_cluster:
    dashboard_address = "128.211.149.133:8787"
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
    "years": ["2018"],
    "syst_variations": ["nominal"],
    # 'syst_variations':['nominal', f'Absolute2016_up'],
    "channels": ["vbf", "vbf_01j", "vbf_2j"],
    # 'channels': ['ggh_01j', 'ggh_2j'],
    "regions": ["h-peak", "h-sidebands"],
}

parameters["mva_bins"] = mva_bins

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
    datasets = grouping.keys()

    parameters["hist_vars"] = ["dimuon_mass"]
    parameters["hist_vars"] += ["score_" + m for m in parameters["dnn_models"]]
    parameters["hist_vars"] += ["score_" + m for m in parameters["bdt_models"]]

    # parameters['plot_vars'] = ['dimuon_mass']
    parameters["plot_vars"] = parameters["hist_vars"]
    parameters["datasets"] = datasets

    paths_grouped = {}
    paths = []
    for y in parameters["years"]:
        for group, ds in grouping_alt.items():
            paths_grouped[group] = []
            for d in ds:
                if d not in datasets:
                    continue
                path = glob.glob(
                    f"{parameters['path']}/"
                    f"{y}_{parameters['label']}/"
                    f"{d}/*.parquet"
                )
                paths.append(path)
                paths_grouped[group].append(path)

    if remake_hists:
        if grouped:
            for group, g_paths in tqdm.tqdm(paths_grouped.items()):
                if len(g_paths) == 0:
                    continue
                workflow(client, g_paths, parameters, timer)
        else:
            workflow(client, paths, parameters, timer)
    plotter(client, parameters, timer)
    timer.summary()
