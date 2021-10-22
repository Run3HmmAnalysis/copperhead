import os
from functools import partial

import dask.dataframe as dd
import pandas as pd
import numpy as np
from hist import Hist
from delphes.config.variables import variables_lookup, Variable
from python.utils import load_from_parquet
from python.utils import save_hist
import uproot3
from uproot3_methods.classes.TH1 import from_numpy

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


def to_templates(client, parameters, hist_df=None):
    # Load saved histograms
    if hist_df is None:
        argsets = []
        for year in parameters["years"]:
            for var_name in parameters["hist_vars"]:
                for dataset in parameters["datasets"]:
                    argsets.append(
                        {"year": year, "var_name": var_name, "dataset": dataset}
                    )
        hist_futures = client.map(
            partial(load_histograms, parameters=parameters), argsets
        )
        hist_rows = client.gather(hist_futures)
        hist_df = pd.concat(hist_rows).reset_index(drop=True)

    hists_to_convert = []
    for year in parameters["years"]:
        for var_name in hist_df.var_name.unique():
            if var_name not in parameters["plot_vars"]:
                continue
            hists_to_convert.append(
                hist_df.loc[(hist_df.var_name == var_name) & (hist_df.year == year)]
            )

    temp_futures = client.map(
        partial(make_templates, parameters=parameters), hists_to_convert
    )
    yields = client.gather(temp_futures)
    return yields


def make_templates(hist, parameters={}):
    if hist.shape[0] == 0:
        return
    var = hist["hist"].values[0].axes[-1]

    # temporary
    region = "h-peak"
    channel = "vbf"
    years = hist.year.unique()
    if len(years) > 1:
        print(
            f"Histograms for more than one year provided. Will make plots only for {years[0]}."
        )
    year = years[0]
    if parameters["save_templates"]:
        path = parameters["templates_path"]
    else:
        path = "/tmp/"

    total_yield = 0
    for dataset in hist.dataset.unique():
        out_fn = f"{path}/{dataset}_{var.name}_{year}.root"
        out_file = uproot3.recreate(out_fn)
        myhist = hist.loc[hist.dataset == dataset, "hist"].values[0]
        the_hist = myhist[region, channel, "value", :].project(var.name).values()
        the_sumw2 = myhist[region, channel, "sumw2", :].project(var.name).values()
        edges = myhist[region, channel, "value", :].project(var.name).axes[0].edges
        centers = (edges[:-1] + edges[1:]) / 2.0

        name = f"{dataset}_{region}_{channel}"
        th1_data = from_numpy([the_hist, edges])
        th1_data._fName = name
        th1_data._fSumw2 = np.array(the_sumw2)
        th1_data._fTsumw2 = np.array(the_sumw2).sum()
        th1_data._fTsumwx2 = np.array(the_sumw2 * centers).sum()
        out_file[name] = th1_data
        out_file.close()
        total_yield += the_hist.sum()
    return total_yield
