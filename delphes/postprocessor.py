import os
from functools import partial

import dask.dataframe as dd
import pandas as pd
from hist import Hist
from delphes.config.variables import variables_lookup, Variable
from python.utils import load_from_parquet
from python.utils import save_hist

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
pd.options.mode.chained_assignment = None


def workflow(client, paths, parameters, timer=None):
    # Load dataframes
    df_future = client.map(load_from_parquet, paths)
    df_future = client.gather(df_future)
    if timer:
        timer.add_checkpoint("Loaded data from Parquet")

    # Merge dataframes
    try:
        df = dd.concat([d for d in df_future if len(d.columns) > 0])
    except Exception:
        return
    npart = df.npartitions
    df = df.compute()
    df["channel"] = "vbf"
    df.reset_index(inplace=True, drop=True)
    df = dd.from_pandas(df, npartitions=npart)
    if npart > 2 * parameters["ncpus"]:
        df = df.repartition(npartitions=parameters["ncpus"])
    if timer:
        timer.add_checkpoint("Combined into a single Dask DataFrame")

    # temporary
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
    if timer:
        timer.add_checkpoint("Prepared for histogramming")

    argsets = []
    for year in df.year.unique():
        for var_name in parameters["hist_vars"]:
            for dataset in df.dataset.unique():
                argsets.append({"year": year, "var_name": var_name, "dataset": dataset})
    # Make histograms
    hist_futures = client.map(partial(histogram, df=df, parameters=parameters), argsets)
    hist_rows = client.gather(hist_futures)
    hist_df = pd.concat(hist_rows).reset_index(drop=True)
    if timer:
        timer.add_checkpoint("Histogramming")
    return hist_df


def histogram(args, df=pd.DataFrame(), parameters={}):
    year = args["year"]
    var_name = args["var_name"]
    dataset = args["dataset"]
    if var_name in variables_lookup.keys():
        var = variables_lookup[var_name]
    else:
        var = Variable(var_name, var_name, 50, 0, 5)

    regions = parameters["regions"]
    channels = parameters["channels"]

    regions = [r for r in regions if r in df.region.unique()]
    channels = [c for c in channels if c in df.channel.unique()]

    # sometimes different years have different binnings (MVA score)
    if "score" in var.name:
        bins = parameters["mva_bins"][var.name.replace("score_", "")][f"{year}"]
        hist = (
            Hist.new.StrCat(regions, name="region")
            .StrCat(channels, name="channel")
            .StrCat(["value", "sumw2"], name="val_err")
            .Var(bins, name=var.name)
            .Double()
        )
    else:
        hist = (
            Hist.new.StrCat(regions, name="region")
            .StrCat(channels, name="channel")
            .StrCat(["value", "sumw2"], name="val_sumw2")
            .Reg(var.nbins, var.xmin, var.xmax, name=var.name, label=var.caption)
            .Double()
        )

    for region in regions:
        var_name = var.name
        if var.name in df.columns:
            var_name = var.name
        else:
            continue
        for channel in channels:
            slicer = (
                (df.dataset == dataset)
                & (df.region == region)
                & (df.year == year)
                & (df.channel == channel)
            )
            data = df.loc[slicer, var_name]
            weight = df.loc[slicer, "lumi_wgt"] * df.loc[slicer, "mc_wgt"]
            hist.fill(region, channel, "value", data, weight=weight)
            hist.fill(region, channel, "sumw2", data, weight=weight * weight)

    if parameters["save_hists"]:
        save_hist(hist, var.name, dataset, year, parameters)
    hist_row = pd.DataFrame(
        [{"year": year, "var_name": var.name, "dataset": dataset, "hist": hist}]
    )
    return hist_row
