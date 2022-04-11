import dask.dataframe as dd
import pandas as pd

from python.workflow import parallelize
from python.io import (
    load_pandas_from_parquet,
    delete_existing_hists,
    save_dataframe,
)
from stage2.categorizer import split_into_channels
from stage2.mva_evaluators import evaluate_dnn, evaluate_bdt
from stage2.histogrammer import make_histograms

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


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


def process_partitions(client, parameters, df):
    argset = {
        "year": df.year.unique(),
        "dataset": df.dataset.unique(),
    }
    if isinstance(df, pd.DataFrame):
        argset["df"] = [df]
    elif isinstance(df, dd.DataFrame):
        argset["df"] = [(i, df.partitions[i]) for i in range(df.npartitions)]

    delete_existing_hists(df.dataset.unique(), df.year.unique(), parameters)
    hist_info_dfs = parallelize(on_partition, argset, client, parameters)
    hist_info_df_full = pd.concat(hist_info_dfs).reset_index(drop=True)
    return hist_info_df_full


def on_partition(args, parameters):
    year = args["year"]
    dataset = args["dataset"]
    df = args["df"]

    # get partition number, if available
    npart = None
    if isinstance(df, tuple):
        npart = df[0]
        df = df[1]

    # convert from Dask DF to Pandas DF
    if isinstance(df, dd.DataFrame):
        df = df.compute()

    # preprocess
    df.fillna(-999.0, inplace=True)
    df = df[(df.dataset == dataset) & (df.year == year)]
    if "dy_m105_160_amc" in dataset:
        df = df[df.gjj_mass <= 350]
    if "dy_m105_160_vbf_amc" in dataset:
        df = df[df.gjj_mass > 350]

    # < evaluate here MVA scores before categorization, if needed >
    # ...

    # < categorization into channels (ggH, VBF, etc.) >
    split_into_channels(df, v="nominal")
    regions = [r for r in parameters["regions"] if r in df.region.unique()]
    channels = [
        c for c in parameters["channels"] if c in df["channel nominal"].unique()
    ]

    # < evaluate here MVA scores after categorization, if needed >
    syst_variations = parameters.get("syst_variations", ["nominal"])
    dnn_models = parameters.get("dnn_models", {})
    bdt_models = parameters.get("dnn_models", {})
    for v in syst_variations:
        # evaluate Keras DNNs
        for channel, models in dnn_models.items():
            if channel not in parameters["channels"]:
                continue
            for model in models:
                score_name = f"score_{model} {v}"
                df.loc[df[f"channel {v}"] == channel, score_name] = evaluate_dnn(
                    df[df[f"channel {v}"] == channel], v, model, parameters, score_name
                )
        # evaluate XGBoost BDTs
        for channel, models in bdt_models.items():
            if channel not in parameters["channels"]:
                continue
            for model in models:
                score_name = f"score_{model} {v}"
                df.loc[df[f"channel {v}"] == channel, score_name] = evaluate_bdt(
                    df[df[f"channel {v}"] == channel], v, model, parameters, score_name
                )

    # < add secondary categorization / binning here >
    # ...

    # < convert desired columns to histograms >
    # not parallelizing for now - nested parallelism leads to a lock
    hist_info_rows = []
    for var_name in parameters["hist_vars"]:
        hist_info_row = make_histograms(
            df, var_name, year, dataset, regions, channels, npart, parameters
        )
        hist_info_rows.append(hist_info_row)
    hist_info_df = pd.concat(hist_info_rows).reset_index(drop=True)

    # < save desired columns as unbinned data (e.g. dimuon_mass for fits) >
    save_unbinned(df, dataset, year, npart, channels, parameters)

    # < return some info for diagnostics & tests >
    return hist_info_df


def save_unbinned(df, dataset, year, npart, channels, parameters):
    to_save = parameters.get("tosave_unbinned", {})
    for channel, var_names in to_save.items():
        if channel not in channels:
            continue
        vnames = []
        for var in var_names:
            if var in df.columns:
                vnames.append(var)
            elif f"{var} nominal" in df.columns:
                vnames.append(f"{var} nominal")
        save_dataframe(
            df.loc[df["channel nominal"] == channel, vnames],
            channel,
            dataset,
            year,
            parameters,
            npart,
        )
