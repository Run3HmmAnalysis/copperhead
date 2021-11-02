import os

import dask.dataframe as dd
import pandas as pd
from python.io import load_pandas_from_parquet

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
pd.options.mode.chained_assignment = None


def load_dataframe(client, parameters, inputs=[], timer=None):

    if isinstance(inputs, list):
        # Load dataframes
        df_future = client.map(load_pandas_from_parquet, inputs)
        df_future = client.gather(df_future)

        # Merge dataframes
        try:
            df = dd.concat([d for d in df_future if len(d.columns) > 0])
        except Exception:
            return

        npart = df.npartitions
        df = df.compute()
        df.reset_index(inplace=True, drop=True)
        df = dd.from_pandas(df, npartitions=npart)
        if df.npartitions > 2 * parameters["ncpus"]:
            df = df.repartition(npartitions=parameters["ncpus"])

    elif isinstance(inputs, pd.DataFrame):
        df = dd.from_pandas(inputs, npartitions=parameters["ncpus"])

    elif isinstance(inputs, dd.DataFrame):
        df = inputs
        npart = df.npartitions
        df = df.compute()
        df.reset_index(inplace=True, drop=True)
        df = dd.from_pandas(df, npartitions=npart)
        if df.npartitions > 2 * parameters["ncpus"]:
            df = df.repartition(npartitions=parameters["ncpus"])

    else:
        print("Wrong input type:", type(inputs))
        return None

    # temporary
    df["channel"] = "vbf"
    if ("dataset" not in df.columns) and ("s" in df.columns):
        df["dataset"] = df["s"]
    if ("region" not in df.columns) and ("r" in df.columns):
        df["region"] = df["r"]
    if ("channel" not in df.columns) and ("c" in df.columns):
        df["channel"] = df["c"]

    keep_columns = ["dataset", "year", "region", "channel"]
    keep_columns += [c for c in df.columns if "wgt" in c]
    keep_columns += parameters["hist_vars"]

    df = df[[c for c in keep_columns if c in df.columns]]
    df = df.compute()

    df.dropna(axis=1, inplace=True)
    df.reset_index(inplace=True)

    return df
