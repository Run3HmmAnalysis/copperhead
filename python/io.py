import os
import pandas as pd
import dask.dataframe as dd
import pickle
import glob


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


def save_dask_pandas_to_parquet(output, out_dir):
    from dask.distributed import get_worker

    name = None
    for key, task in get_worker().tasks.items():
        if task.state == "executing":
            name = key[-32:]
    if not name:
        return
    for ds in output.dataset.unique():
        df = output[output.dataset == ds]
        if df.shape[0] == 0:
            return
        mkdir(f"{out_dir}/{ds}")
        df.to_parquet(path=f"{out_dir}/{ds}/{name}.parquet")


def save_spark_pandas_to_parquet(output, out_dir):
    from pyspark import TaskContext

    ctx = TaskContext()
    name = f"part_{ctx.partitionId()}"
    # print("Stage: {0}, Partition: {1}, Host: {2}".format(
    #     ctx.stageId(), ctx.partitionId(), socket.gethostname()))

    for ds in output.dataset.unique():
        df = output[output.dataset == ds]
        if df.shape[0] == 0:
            return
        mkdir(f"{out_dir}/{ds}")
        path = f"{out_dir}/{ds}/{name}.parquet"
        df.to_parquet(path=path)
        print(f"Saved to {path}")


def load_dataframe(client, parameters, inputs=[]):
    if isinstance(inputs, list):
        # Load dataframes
        df_future = client.map(load_pandas_from_parquet, inputs)
        df_future = client.gather(df_future)
        # Merge dataframes
        try:
            df = dd.concat([d for d in df_future if d.shape[1] > 0])
        except Exception:
            return None
        if df.npartitions > 2 * parameters["ncpus"]:
            df = df.repartition(npartitions=parameters["ncpus"])

    elif isinstance(inputs, pd.DataFrame):
        df = dd.from_pandas(inputs, npartitions=parameters["ncpus"])

    elif isinstance(inputs, dd.DataFrame):
        if inputs.npartitions > 2 * parameters["ncpus"]:
            df = inputs.repartition(npartitions=parameters["ncpus"])
        else:
            df = inputs

    else:
        print("Wrong input type:", type(inputs))
        return None

    # for now ignoring systematics
    ignore_columns = [c for c in df.columns if (("wgt_" in c) and ("nominal" not in c))]
    ignore_columns += [c for c in df.columns if "pdf_" in c]
    df = df[[c for c in df.columns if c not in ignore_columns]]

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


def save_histogram(hist, var_name, dataset, year, parameters, npart=None):
    mkdir(parameters["hist_path"])
    hist_path = parameters["hist_path"] + parameters["label"]
    mkdir(hist_path)
    mkdir(f"{hist_path}/{year}")
    mkdir(f"{hist_path}/{year}/{var_name}")
    if npart is None:
        path = f"{hist_path}/{year}/{var_name}/{dataset}.pickle"
    else:
        path = f"{hist_path}/{year}/{var_name}/{dataset}_{npart}.pickle"
    with open(path, "wb") as handle:
        pickle.dump(hist, handle, protocol=pickle.HIGHEST_PROTOCOL)


def delete_existing_hists(datasets, years, parameters):
    for year in years:
        for var_name in parameters["hist_vars"]:
            for dataset in datasets:
                try:
                    hist_path = parameters["hist_path"] + parameters["label"]
                    paths = glob.glob(
                        f"{hist_path}/{year}/{var_name}/{dataset}_*.pickle"
                    ) + [f"{hist_path}/{year}/{var_name}/{dataset}.pickle"]
                    for path in paths:
                        remove(path)
                except Exception:
                    pass


def load_histogram(argset, parameters):
    year = argset["year"]
    var_name = argset["var_name"]
    dataset = argset["dataset"]
    hist_path = parameters["hist_path"] + parameters["label"]
    paths = glob.glob(f"{hist_path}/{year}/{var_name}/{dataset}_*.pickle") + [
        f"{hist_path}/{year}/{var_name}/{dataset}.pickle"
    ]
    hist_df = pd.DataFrame()
    for path in paths:
        try:
            with open(path, "rb") as handle:
                hist = pickle.load(handle)
                hist_df = pd.concat(
                    [
                        hist_df,
                        pd.DataFrame(
                            [
                                {
                                    "year": year,
                                    "var_name": var_name,
                                    "dataset": dataset,
                                    "hist": hist,
                                }
                            ]
                        ),
                    ]
                )
                hist_df.reset_index(drop=True, inplace=True)
        except Exception:
            pass
    return hist_df


def save_dataframe(df, channel, dataset, year, parameters, npart=None):
    mkdir(parameters["unbinned_path"])
    unbin_path = parameters["unbinned_path"] + parameters["label"]
    mkdir(unbin_path)
    mkdir(f"{unbin_path}/{channel}_{year}")
    if npart is None:
        path = f"{unbin_path}/{channel}_{year}/{dataset}.parquet"
    else:
        path = f"{unbin_path}/{channel}_{year}/{dataset}_{npart}.parquet"
    df.to_parquet(path=path)


def save_template(templates, out_name, parameters):
    import uproot3

    out_file = uproot3.recreate(out_name)
    for tmp in templates:
        out_file[tmp._fName] = tmp
    out_file.close()
    return
