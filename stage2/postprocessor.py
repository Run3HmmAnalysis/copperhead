import dask.dataframe as dd
import pandas as pd

from python.workflow import parallelize
from python.io import (
    delete_existing_stage2_hists,
    delete_existing_stage2_parquet,
    save_stage2_output_parquet,
)
from stage2.categorizer import split_into_channels
from stage2.mva_evaluators import evaluate_pytorch_dnn, evaluate_bdt
from stage2.histogrammer import make_histograms

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


def process_partitions(client, parameters, df):
    # for now ignoring some systematics
    ignore_columns = []
    ignore_columns += [c for c in df.columns if "pdf_" in c]

    df = df[[c for c in df.columns if c not in ignore_columns]]

    years = df.year.unique()
    datasets = df.dataset.unique()
    # delete previously generated outputs to prevent partial overwrite
    delete_existing_stage2_hists(datasets, years, parameters)
    delete_existing_stage2_parquet(datasets, years, parameters)

    # prepare parameters for parallelization
    argset = {
        "year": years,
        "dataset": datasets,
    }
    if isinstance(df, pd.DataFrame):
        argset["df"] = [df]
    elif isinstance(df, dd.DataFrame):
        argset["df"] = [(i, df.partitions[i]) for i in range(df.npartitions)]

    # perform categorization, evaluate mva models, fill histograms
    hist_info_dfs = parallelize(on_partition, argset, client, parameters)

    # return info for debugging
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
        c for c in parameters["channels"] if c in df["channel_nominal"].unique()
    ]

    # < evaluate here MVA scores after categorization, if needed >
    syst_variations = parameters.get("syst_variations", ["nominal"])
    dnn_models = parameters.get("dnn_models", {})
    bdt_models = parameters.get("bdt_models", {})
    for v in syst_variations:
        # evaluate Keras DNNs
        for channel, models in dnn_models.items():
            if channel not in parameters["channels"]:
                continue
            for model in models:
                score_name = f"score_{model}_{v}"
                df.loc[
                    df[f"channel_{v}"] == channel, score_name
                ] = evaluate_pytorch_dnn(
                    df[df[f"channel_{v}"] == channel],
                    v,
                    model,
                    parameters,
                    score_name,
                    channel,
                )

        # evaluate XGBoost BDTs
        for channel, models in bdt_models.items():
            if channel not in parameters["channels"]:
                continue
            for model in models:
                score_name = f"score_{model}_{v}"
                df.loc[df[f"channel_{v}"] == channel, score_name] = evaluate_bdt(
                    df[df[f"channel_{v}"] == channel], v, model, parameters, score_name
                )

    # < add secondary categorization / binning here >
    # ...

    # temporary implementation: move from mva score to mva bin number
    if "score_pytorch_test_nominal" in df.columns:
        mva_bins = parameters["mva_bins_original"]["pytorch_test"][str(year)]
        for i in range(len(mva_bins) - 1):
            lo = mva_bins[i]
            hi = mva_bins[i + 1]
            cut = (df["score_pytorch_test_nominal"] > lo) & (
                df["score_pytorch_test_nominal"] <= hi
            )
            df.loc[cut, "bin_number"] = i
        df["score_pytorch_test_nominal"] = df["bin_number"]
        parameters["mva_bins"] = {
            "pytorch_test": {
                "2016": list(range(len(mva_bins))),
                "2017": list(range(len(mva_bins))),
                "2018": list(range(len(mva_bins))),
            }
        }

    # < convert desired columns to histograms >
    # not parallelizing for now - nested parallelism leads to a lock
    hist_info_rows = []
    for var_name in parameters["hist_vars"]:
        hist_info_row = make_histograms(
            df, var_name, year, dataset, regions, channels, npart, parameters
        )
        if hist_info_row is not None:
            hist_info_rows.append(hist_info_row)
    if len(hist_info_rows) == 0:
        return pd.DataFrame()

    hist_info_df = pd.concat(hist_info_rows).reset_index(drop=True)

    # < save desired columns as unbinned data (e.g. dimuon_mass for fits) >
    do_save_unbinned = parameters.get("save_unbinned", False)
    if do_save_unbinned:
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
            elif f"{var}_nominal" in df.columns:
                vnames.append(f"{var}_nominal")
        save_stage2_output_parquet(
            df.loc[df["channel_nominal"] == channel, vnames],
            channel,
            dataset,
            year,
            parameters,
            npart,
        )
