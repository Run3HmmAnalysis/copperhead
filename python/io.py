import os
import pandas as pd
import dask.dataframe as dd
import pickle


def mkdir(path):
    try:
        os.mkdir(path)
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


def load_pandas_from_parquet(path):
    df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    if len(path) > 0:
        try:
            df = dd.read_parquet(path)
        except Exception:
            return df
    return df


def save_histogram(hist, var_name, dataset, year, parameters):
    mkdir(parameters["hist_path"])
    hist_path = parameters["hist_path"] + parameters["label"]
    mkdir(hist_path)
    mkdir(f"{hist_path}/{year}")
    mkdir(f"{hist_path}/{year}/{var_name}")
    path = f"{hist_path}/{year}/{var_name}/{dataset}.pickle"
    with open(path, "wb") as handle:
        pickle.dump(hist, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_histogram(argset, parameters):
    year = argset["year"]
    var_name = argset["var_name"]
    dataset = argset["dataset"]
    hist_path = parameters["hist_path"] + parameters["label"]
    path = f"{hist_path}/{year}/{var_name}/{dataset}.pickle"
    try:
        with open(path, "rb") as handle:
            hist = pickle.load(handle)
    except Exception:
        return pd.DataFrame()
    hist_row = pd.DataFrame(
        [{"year": year, "var_name": var_name, "dataset": dataset, "hist": hist}]
    )
    return hist_row


def save_template(templates, out_name, parameters):
    import uproot3

    out_file = uproot3.recreate(out_name)
    for tmp in templates:
        out_file[tmp._fName] = tmp
    out_file.close()
    return
