import glob
import tqdm
import argparse
import dask
from dask.distributed import Client

from python.timer import Timer
from delphes.postprocessor import workflow
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
    "label": "oct14",
    "path": "/depot/cms/hmm/coffea/",
    "hist_path": "/depot/cms/hmm/coffea/snowmass_histograms/",
    "plots_path": "./plots_test/snowmass/",
    "years": ["snowmass"],
    "syst_variations": ["nominal"],
    "channels": ["vbf", "vbf_01j", "vbf_2j"],
    # 'channels': ['ggh_01j', 'ggh_2j'],
    "regions": ["h-peak", "h-sidebands"],
    "save_hists": True,
    "save_plots": True,
    "plot_ratio": False,
    "14TeV_label": True,
    "has_variations": False,
}

parameters["grouping"] = {
    # "ggh_powheg": "ggH",
    # "vbf_powheg": "VBF",
    "dy_m100_mg": "DY",
    "ttbar_dl": "TTbar",
    "tttj": "TTbar",
    "tttt": "TTbar",
    "tttw": "TTbar",
    "ttwj": "TTbar",
    "ttww": "TTbar",
    "ttz": "TTbar",
    "st_s": "Single top",
    # "st_t_top": "Single top",
    "st_t_antitop": "Single top",
    "st_tw_top": "Single top",
    "st_tw_antitop": "Single top",
    "zz_2l2q": "VV",
}

grouping_alt = {
    "DY": ["dy_m100_mg"],
    # "EWK": [],
    "TTbar": ["ttbar_dl", "tttj", "tttt", "tttw", "ttwj", "ttww", "ttz"],
    "Single top": ["st_s", "st_t_antitop", "st_tw_top", "st_tw_antitop"],
    "VV": ["zz_2l2q"],
    # "VVV": [],
    # "ggH": ["ggh_amcPS"],
    # "VBF": ["vbf_powheg_dipole"],
}

parameters["plot_groups"] = {
    "stack": ["DY", "EWK", "TTbar", "Single top", "VV", "VVV"],
    "step": ["VBF", "ggH"],
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

    datasets = parameters["grouping"].keys()
    # datasets = ["dy_m100_mg"]

    parameters["hist_vars"] = [
        "dimuon_mass",
        "dimuon_pt",
        "dimuon_eta",
        "dimuon_phi",
        "dimuon_dEta",
        "dimuon_dPhi",
        "dimuon_dR",
        "dimuon_rap",
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
        "jet1_rap",
        "jet1_phi",
        "jet2_pt",
        "jet2_eta",
        "jet2_rap",
        "jet2_phi",
        "jj_mass",
        "jj_pt",
        "jj_eta",
        "jj_phi",
        "jj_dEta",
        "jj_dPhi",
        "mmj1_dEta",
        "mmj1_dPhi",
        "mmj1_dR",
        "mmj2_dEta",
        "mmj2_dPhi",
        "mmj2_dR",
        "mmj_min_dEta",
        "mmj_min_dPhi",
        "mmjj_pt",
        "mmjj_eta",
        "mmjj_phi",
        "mmjj_mass",
        "rpt",
        "zeppenfeld",
        "ll_zstar_log",
        "njets",
    ]

    parameters["plot_vars"] = parameters["hist_vars"]
    parameters["datasets"] = datasets

    how = {
        "DY": "all",
        # "EWK": "all",
        "TTbar": "individual",
        "Single top": "individual",
        "VV": "individual",
        "VVV": "individual",
        # "ggH": "all",
        # "VBF": "all",
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
                workflow(client, [path], parameters, timer=timer)
        else:
            for year, groups in paths_grouped.items():
                print(f"Processing {year}")
                for group, g_paths in tqdm.tqdm(groups.items()):
                    if len(g_paths) == 0:
                        continue
                    workflow(client, g_paths, parameters, timer=timer)

    if args.plot:
        plotter(client, parameters, timer=timer)
    timer.summary()
