import os
import pandas as pd
import dask.dataframe as dd
from dask.distributed import get_worker
import pickle
import glob
import uproot3


def mkdir(path):
    try:
        os.mkdir(path)
    except Exception:
        pass


def remove(path):
    try:
        os.remove(path)
    except Exception:
        pass


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
        mkdir(f"{out_dir}/{dataset}")
        df.to_parquet(path=f"{out_dir}/{dataset}/{name}.parquet")


def load_dataframe(client, parameters, inputs=[], dataset=None):
    ncpus = parameters.get("ncpus", 1)
    custom_npartitions_dict = parameters.get("custom_npartitions", {})
    custom_npartitions = 0
    if dataset in custom_npartitions_dict.keys():
        custom_npartitions = custom_npartitions_dict[dataset]

    if isinstance(inputs, list):
        # Load dataframes
        df_future = client.map(load_pandas_from_parquet, inputs)
        df_future = client.gather(df_future)
        # Merge dataframes
        try:
            df = dd.concat([d for d in df_future if d.shape[1] > 0])
        except Exception:
            return None
        if custom_npartitions > 0:
            df = df.repartition(npartitions=custom_npartitions)
        elif df.npartitions > 2 * ncpus:
            df = df.repartition(npartitions=ncpus)

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
    hist_path = parameters.get("hist_path", None)
    label = parameters.get("label", None)
    if (hist_path is None) or (label is None):
        return
    hist_path_full = hist_path + "/" + label

    mkdir(hist_path)
    mkdir(hist_path_full)
    mkdir(f"{hist_path_full}/{year}")
    mkdir(f"{hist_path_full}/{year}/{var_name}")
    if npart is None:
        path = f"{hist_path_full}/{year}/{var_name}/{dataset}.pickle"
    else:
        path = f"{hist_path_full}/{year}/{var_name}/{dataset}_{npart}.pickle"
    with open(path, "wb") as handle:
        pickle.dump(hist, handle, protocol=pickle.HIGHEST_PROTOCOL)


def delete_existing_stage2_hists(datasets, years, parameters):
    var_names = parameters.get("hist_vars", [])
    hist_path = parameters.get("hist_path", None)
    label = parameters.get("label", None)
    if (hist_path is None) or (label is None):
        return
    hist_path_full = hist_path + "/" + label

    for year in years:
        for var_name in var_names:
            for dataset in datasets:
                try:
                    paths = glob.glob(
                        f"{hist_path_full}/{year}/{var_name}/{dataset}_*.pickle"
                    ) + glob.glob(
                        f"{hist_path_full}/{year}/{var_name}/{dataset}.pickle"
                    )
                    for file in paths:
                        remove(file)
                except Exception:
                    pass


def load_stage2_output_hists(argset, parameters):
    year = argset["year"]
    var_name = argset["var_name"]
    dataset = argset["dataset"]
    hist_path = parameters.get("hist_path", None)
    label = parameters.get("label", None)
    if (hist_path is None) or (label is None):
        return
    hist_path_full = hist_path + "/" + label

    paths = glob.glob(
        f"{hist_path_full}/{year}/{var_name}/{dataset}_*.pickle"
    ) + glob.glob(f"{hist_path_full}/{year}/{var_name}/{dataset}.pickle")
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
    stage2_parquet_path = parameters.get("stage2_parquet_path", None)
    label = parameters.get("label", None)
    if (stage2_parquet_path is None) or (label is None):
        return
    path_full = stage2_parquet_path + "/" + label

    mkdir(stage2_parquet_path)
    mkdir(path_full)
    mkdir(f"{path_full}/{channel}_{year}")
    if npart is None:
        path = f"{path_full}/{channel}_{year}/{dataset}.parquet"
    else:
        path = f"{path_full}/{channel}_{year}/{dataset}_{npart}.parquet"
    df.to_parquet(path=path)


def delete_existing_stage2_parquet(datasets, years, parameters):
    to_delete = parameters.get("tosave_unbinned", {})
    stage2_parquet_path = parameters.get("stage2_parquet_path", None)
    label = parameters.get("label", None)
    if (stage2_parquet_path is None) or (label is None):
        return
    path_full = stage2_parquet_path + "/" + label

    for channel in to_delete.keys():
        for year in years:
            for dataset in datasets:
                paths = glob.glob(
                    f"{path_full}/{channel}_{year}/{dataset}_*.parquet"
                ) + glob.glob(f"{path_full}/{channel}_{year}/{dataset}.parquet")
                for file in paths:
                    remove(file)


def save_template(templates, out_name, parameters):
    out_file = uproot3.recreate(out_name)
    for tmp in templates:
        out_file[tmp._fName] = tmp
    out_file.close()
    return
