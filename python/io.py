import os
import pandas as pd
import dask.dataframe as dd
from dask.distributed import get_worker
import pickle
import glob
import re
import uproot3
import subprocess
import traceback

EOS_SERVER = "davs://eos.cms.rcac.purdue.edu:9000/"


# new implementation (for eos)
def mkdir(path):
    eos_path = EOS_SERVER + path
    print(f"Creating directory {eos_path}")
    try:
        subprocess.run([f"gfal-mkdir {eos_path}"], shell=True)
    except Exception as e:
        tb = traceback.format_exc()
        return "Failed: " + str(e) + " " + tb


def remove(path):
    eos_path = EOS_SERVER + path
    try:
        subprocess.run([f"gfal-rm -r {eos_path}"], shell=True)
    except Exception as e:
        tb = traceback.format_exc()
        return "Failed: " + str(e) + " " + tb


def save_stage1_output_to_parquet(output, out_dir):
    name = None
    for key, task in get_worker().tasks.items():
        if task.state == "executing":
            name = key[-32:]
    if not name:
        return
    for dataset in output.dataset.unique():
        df = output[output.dataset == dataset]
        if df.shape[0] == 0:
            return

        path = f"{out_dir}/{dataset}"
        print(f"Create {path}?")
        mkdir(path)

        df.to_parquet(path=f"/tmp/{name}.parquet")

        eos_path = EOS_SERVER + path
        subprocess.run(
            [f"gfal-copy file:////tmp/{name}.parquet {eos_path}"], shell=True
        )
        subprocess.run([f"rm /tmp/{name}.parquet"], shell=True)


def delete_existing_stage1_output(datasets, parameters):
    global_path = parameters.get("global_path", None)
    label = parameters.get("label", None)
    year = parameters.get("year", None)

    if (global_path is None) or (label is None) or (year is None):
        return

    for dataset in datasets:
        path = f"{global_path}/{label}/stage1_output/{year}/{dataset}/"
        eos_path = EOS_SERVER + path
        print(f"Deleting files in {eos_path}")
        try:
            paths = subprocess.run(
                [f"gfal-ls {eos_path}"], stdout=subprocess.PIPE, shell=True
            ).stdout.splitlines()
            for file in paths:
                file = file.decode("utf-8").rstrip("\r|\n")
                subprocess.run([f"gfal-rm {eos_path}/{file}"], shell=True)
        except Exception as e:
            tb = traceback.format_exc()
            return "Failed: " + str(e) + " " + tb


# old implementation (for hadoop)
def mkdir_(path):
    try:
        os.mkdir(path)
    except Exception:
        pass


def remove_(path):
    try:
        os.remove(path)
    except Exception:
        pass


def save_stage1_output_to_parquet_(output, out_dir):
    name = None
    for key, task in get_worker().tasks.items():
        if task.state == "executing":
            name = key[-32:]
    if not name:
        return
    for dataset in output.dataset.unique():
        df = output[output.dataset == dataset]
        if df.shape[0] == 0:
            return
        mkdir(f"{out_dir}/{dataset}")
        df.to_parquet(path=f"{out_dir}/{dataset}/{name}.parquet")


def delete_existing_stage1_output_(datasets, parameters):
    global_path = parameters.get("global_path", None)
    label = parameters.get("label", None)
    year = parameters.get("year", None)

    if (global_path is None) or (label is None) or (year is None):
        return

    for dataset in datasets:
        path = f"{global_path}/{label}/stage1_output/{year}/{dataset}/"
        paths = glob.glob(f"{path}/*.parquet")
        for file in paths:
            remove(file)


def load_dataframe(client, parameters, inputs=[], dataset=None):
    ncpus = parameters.get("ncpus", 1)
    custom_npartitions_dict = parameters.get("custom_npartitions", {})
    custom_npartitions = 0
    if dataset in custom_npartitions_dict.keys():
        custom_npartitions = custom_npartitions_dict[dataset]

    if isinstance(inputs, list):
        # Load dataframes
        if client:
            df_future = client.map(load_pandas_from_parquet, inputs)
            df_future = client.gather(df_future)
        else:
            df_future = []
            for inp in inputs:
                df_future.append(load_pandas_from_parquet(inp))
        # Merge dataframes
        try:
            df = dd.concat([d for d in df_future if d.shape[1] > 0])
        except Exception:
            return None
        if custom_npartitions > 0:
            df = df.repartition(npartitions=custom_npartitions)
        elif df.npartitions > 2 * ncpus:
            df = df.repartition(npartitions=2 * ncpus)

    elif isinstance(inputs, pd.DataFrame):
        df = dd.from_pandas(inputs, npartitions=ncpus)

    elif isinstance(inputs, dd.DataFrame):
        if custom_npartitions > 0:
            df = inputs.repartition(npartitions=custom_npartitions)
        elif inputs.npartitions > 2 * ncpus:
            df = inputs.repartition(npartitions=ncpus)
        else:
            df = inputs

    else:
        print("Wrong input type:", type(inputs))
        return None

    return df


def load_pandas_from_parquet(path):
    df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    df = dd.read_parquet(path)
    if len(path) > 0:
        try:
            df = dd.read_parquet(path)
        except Exception:
            return df
    return df


def save_stage2_output_hists(hist, var_name, dataset, year, parameters, npart=None):
    global_path = parameters.get("global_path", None)
    label = parameters.get("label", None)

    if (global_path is None) or (label is None):
        return

    out_dir = global_path + "/" + label
    mkdir(out_dir)
    out_dir += "/" + "stage2_histograms"
    mkdir(out_dir)
    out_dir += "/" + var_name
    mkdir(out_dir)
    out_dir += "/" + str(year)
    mkdir(out_dir)

    if npart is None:
        path = f"{out_dir}/{dataset}.pickle"
    else:
        path = f"{out_dir}/{dataset}_{npart}.pickle"

    with open(path, "wb") as handle:
        pickle.dump(hist, handle, protocol=pickle.HIGHEST_PROTOCOL)


def delete_existing_stage2_hists(datasets, years, parameters):
    var_names = parameters.get("hist_vars", [])
    global_path = parameters.get("global_path", None)
    label = parameters.get("label", None)

    if (global_path is None) or (label is None):
        return

    for year in years:
        for var_name in var_names:
            for dataset in datasets:
                path = f"{global_path}/{label}/stage2_histograms/{var_name}/{year}/"
                try:
                    paths = [f"{path}/{dataset}.pickle"]
                    for fname in os.listdir(path):
                        if re.fullmatch(rf"{dataset}_[0-9]+.pickle", fname):
                            paths.append(f"{path}/{fname}")
                    for file in paths:
                        remove(file)
                except Exception:
                    pass


def load_stage2_output_hists(argset, parameters):
    year = argset["year"]
    var_name = argset["var_name"]
    dataset = argset["dataset"]
    global_path = parameters.get("global_path", None)
    label = parameters.get("label", None)

    if (global_path is None) or (label is None):
        return

    path = f"{global_path}/{label}/stage2_histograms/{var_name}/{year}/"
    paths = [f"{path}/{dataset}.pickle"]
    for fname in os.listdir(path):
        if re.fullmatch(rf"{dataset}_[0-9]+.pickle", fname):
            paths.append(f"{path}/{fname}")

    hist_df = pd.DataFrame()
    for path in paths:
        try:
            with open(path, "rb") as handle:
                hist = pickle.load(handle)
                new_row = {
                    "year": year,
                    "var_name": var_name,
                    "dataset": dataset,
                    "hist": hist,
                }
                hist_df = pd.concat([hist_df, pd.DataFrame([new_row])])
                hist_df.reset_index(drop=True, inplace=True)
        except Exception:
            pass
    return hist_df


def save_stage2_output_parquet(df, channel, dataset, year, parameters, npart=None):
    global_path = parameters.get("global_path", None)
    label = parameters.get("label", None)
    if (global_path is None) or (label is None):
        return

    out_dir = global_path + "/" + label
    mkdir(out_dir)
    out_dir += "/" + "stage2_unbinned"
    mkdir(out_dir)
    out_dir += "/" + f"{channel}_{year}"
    mkdir(out_dir)

    if npart is None:
        path = f"{out_dir}/{dataset}.parquet"
    else:
        path = f"{out_dir}/{dataset}_{npart}.parquet"
    df.to_parquet(path=path)


def delete_existing_stage2_parquet(datasets, years, parameters):
    to_delete = parameters.get("tosave_unbinned", {})
    global_path = parameters.get("global_path", None)
    label = parameters.get("label", None)
    if (global_path is None) or (label is None):
        return

    for channel in to_delete.keys():
        for year in years:
            for dataset in datasets:
                path = f"{global_path}/{label}/stage2_unbinned/{channel}_{year}/"
                paths = [f"{path}/{dataset}.parquet"]
                for fname in os.listdir(path):
                    if re.fullmatch(rf"{dataset}_[0-9]+.parquet", fname):
                        paths.append(f"{path}/{fname}")
                for file in paths:
                    remove(file)


def save_template(templates, out_name, parameters):
    out_file = uproot3.recreate(out_name)
    for tmp in templates:
        out_file[tmp._fName] = tmp
    out_file.close()
    return
