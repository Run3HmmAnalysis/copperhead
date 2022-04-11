import glob
import tqdm
import argparse
import dask
from dask.distributed import Client
import dask.dataframe as dd

from python.io import load_dataframe
from stage2.postprocessor import process_partitions
from config.mva_bins import mva_bins
from config.variables import variables_lookup

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
    "path": "/depot/cms/hmm/coffea/",
    "hist_path": "/depot/cms/hmm/coffea/histograms/",
    "models_path": "/depot/cms/hmm/trained_models/",
    "dnn_models": {
        # "vbf": ["dnn_allyears_128_64_32"],
    },
    "bdt_models": {},
    "years": args.years,
    "syst_variations": ["nominal"],
    # 'syst_variations':['nominal', 'Absolute2016_up'],
    # "channels": ["vbf"],
    "channels": ["ggh_0jets"],
    "regions": ["h-peak", "h-sidebands"],
    "save_hists": True,
    "variables_lookup": variables_lookup,
    "unbinned_path": "/depot/cms/hmm/coffea/stage2_unbinned/",
    "tosave_unbinned": {
        "ggh_0jets": ["dimuon_mass", "wgt_nominal"],
        "ggh_1jet": ["dimuon_mass", "wgt_nominal"],
        "ggh_2orMoreJets": ["dimuon_mass", "wgt_nominal"],
    },
}

parameters["mva_bins"] = mva_bins
parameters["datasets"] = [
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
    "ewk_lljj_mll105_160_py_dipole",
    "ttjets_dl",
    "ttjets_sl",
    "ttw",
    "ttz",
    "st_tw_top",
    "st_tw_antitop",
    "ww_2l2nu",
    "wz_2l2q",
    "wz_1l1nu2q",
    "wz_3lnu",
    "zz",
    "www",
    "wwz",
    "wzz",
    "zzz",
    "ggh_amcPS",
    "vbf_powheg_dipole",
]
parameters["datasets"] = ["vbf_powheg_dipole"]

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

    parameters["hist_vars"] = ["dimuon_mass"]
    for models in list(parameters["dnn_models"].values()) + list(
        parameters["bdt_models"].values()
    ):
        for model in models:
            parameters["hist_vars"] += ["score_" + model]

    # prepare lists of paths to parquet files for each year and dataset
    all_paths = {}
    for year in parameters["years"]:
        all_paths[year] = {}
        for dataset in parameters["datasets"]:
            paths = glob.glob(
                f"{parameters['path']}/"
                f"{year}_{parameters['label']}/"
                f"{dataset}/*.parquet"
            )
            all_paths[year][dataset] = paths

    for year in parameters["years"]:
        print(f"Processing {year}")
        for dataset, path in tqdm.tqdm(all_paths[year].items()):
            if len(path) == 0:
                continue
            df = load_dataframe(client, parameters, inputs=[path])
            if not isinstance(df, dd.DataFrame):
                continue
            info = process_partitions(client, parameters, df)
