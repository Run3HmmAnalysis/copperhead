import os
import pandas as pd
import dask.dataframe as dd
import pickle


def almost_equal(a, b):
    return abs(a - b) < 10e-6


def mkdir(path):
    try:
        os.mkdir(path)
    except Exception:
        pass


def load_from_parquet(path):
    df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    if len(path) > 0:
        try:
            df = dd.read_parquet(path)
        except Exception:
            return df
    return df


def save_hist(hist, var_name, dataset, year, parameters):
    mkdir(parameters["hist_path"])
    hist_path = parameters["hist_path"] + parameters["label"]
    mkdir(hist_path)
    mkdir(f"{hist_path}/{year}")
    mkdir(f"{hist_path}/{year}/{var_name}")
    path = f"{hist_path}/{year}/{var_name}/{dataset}.pickle"
    with open(path, "wb") as handle:
        pickle.dump(hist, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_histograms(argset, parameters):
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
