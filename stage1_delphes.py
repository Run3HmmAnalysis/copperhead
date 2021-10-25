import time
import argparse
import traceback
from functools import partial

from coffea.nanoevents import DelphesSchema
from coffea.processor import dask_executor, run_uproot_job

from python.utils import mkdir
from python.io import save_dask_pandas_to_parquet
from delphes.preprocessor import get_fileset
from delphes.processor import DimuonProcessorDelphes
from delphes.config.datasets import datasets

from dask.distributed import Client

# dask.config.set({"temporary-directory": "/depot/cms/hmm/dask-temp/"})

parser = argparse.ArgumentParser()
# Slurm cluster IP to use. If not specified, will create a local cluster
parser.add_argument(
    "-sl",
    "--slurm",
    dest="slurm_port",
    default=None,
    action="store",
    help="Slurm cluster port (if not specified, " "will create a local cluster)",
)
parser.add_argument(
    "-l",
    "--label",
    dest="label",
    default="test",
    action="store",
    help="Unique run label (to create output path)",
)
parser.add_argument(
    "-ch",
    "--chunksize",
    dest="chunksize",
    default=100000,
    action="store",
    help="Approximate chunk size",
)
parser.add_argument(
    "-mch",
    "--maxchunks",
    dest="maxchunks",
    default=-1,
    action="store",
    help="Max. number of chunks",
)

args = parser.parse_args()

node_ip = "128.211.149.133"  # hammer-c000
# node_ip = '128.211.149.140' # hammer-c007
dash_local = f"{node_ip}:34875"

if args.slurm_port is None:
    local_cluster = True
    slurm_cluster_ip = ""
else:
    local_cluster = False
    slurm_cluster_ip = f"{node_ip}:{args.slurm_port}"

mch = None if int(args.maxchunks) < 0 else int(args.maxchunks)

year = "snowmass"

parameters = {
    "year": year,
    "label": args.label,
    "global_out_path": "/depot/cms/hmm/coffea/",
    "out_path": f"{year}_{args.label}",
    "server": "/mnt/hadoop/",
    "chunksize": int(args.chunksize),
    "maxchunks": mch,
    "local_cluster": local_cluster,
    "slurm_cluster_ip": slurm_cluster_ip,
    "lumi": 3000000.0,
}

parameters["out_dir"] = f"{parameters['global_out_path']}/{parameters['out_path']}"


def submit_job(client, parameters):
    mkdir(parameters["out_dir"])
    out_dir = f"{parameters['out_dir']}/"
    mkdir(out_dir)

    executor = dask_executor
    executor_args = {"client": client, "schema": DelphesSchema, "retries": 0}
    processor_args = {
        "apply_to_output": partial(save_dask_pandas_to_parquet, out_dir=out_dir)
    }

    try:
        run_uproot_job(
            parameters["fileset"],
            "Delphes",
            DimuonProcessorDelphes(**processor_args),
            executor,
            executor_args=executor_args,
            chunksize=parameters["chunksize"],
            maxchunks=parameters["maxchunks"],
        )
    except Exception as e:
        tb = traceback.format_exc()
        return "Failed: " + str(e) + " " + tb

    return "Success!"


if __name__ == "__main__":
    tick = time.time()
    if parameters["local_cluster"]:
        client = Client(
            processes=True,
            n_workers=40,
            dashboard_address=dash_local,
            threads_per_worker=1,
            memory_limit="2.9GB",
        )
    else:
        client = Client(parameters["slurm_cluster_ip"])
    print("Client created")

    ds_names = [
        # "ggh_powheg",
        # "vbf_powheg",
        "dy_m100_mg",
        "ttbar_dl",
        "tttj",
        "tttt",
        "tttw",
        "ttwj",
        "ttww",
        "ttz",
        "st_s",
        # "st_t_top", # hangs
        "st_t_antitop",
        "st_tw_top",
        "st_tw_antitop",
        "zz_2l2q",
    ]
    # ds_names = ["dy_m100_mg"]
    my_datasets = {name: path for name, path in datasets.items() if name in ds_names}
    fileset_json = "/depot/cms/hmm/coffea/snowmass_datasets.json"
    fileset = get_fileset(
        my_datasets,
        parameters,
        save_to=fileset_json,
        # load_from=fileset_json,
    )

    # Process all datasets at once
    # parameters["fileset"] = fileset
    # out = submit_job({}, parameters)
    # print(out)

    # for name, data in fileset.items():
    #     print(name, data["metadata"]["lumi_wgt"])
    # import sys
    # sys.exit()

    # Process datasets individually
    for name, data in fileset.items():
        if name not in ds_names:
            continue
        parameters["fileset"] = {name: data}
        out = submit_job(client, parameters)
        print(name, out)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")
