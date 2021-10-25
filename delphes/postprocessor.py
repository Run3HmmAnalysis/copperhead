import os
from functools import partial

import dask.dataframe as dd
import pandas as pd
import numpy as np
from hist import Hist
from delphes.config.variables import variables_lookup, Variable
from python.io import load_pandas_from_parquet
from python.io import save_histogram, load_histogram
from python.io import save_template

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from uproot3_methods.classes.TH1 import from_numpy

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


def to_histograms(client, parameters, df):
    argsets = []
    for year in df.year.unique():
        for var_name in parameters["hist_vars"]:
            for dataset in df.dataset.unique():
                argsets.append({"year": year, "var_name": var_name, "dataset": dataset})

    hist_futures = client.map(
        partial(make_histograms, df=df, parameters=parameters), argsets
    )
    hist_rows = client.gather(hist_futures)
    hist_df = pd.concat(hist_rows).reset_index(drop=True)

    return hist_df


def make_histograms(args, df=pd.DataFrame(), parameters={}):
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
        save_histogram(hist, var.name, dataset, year, parameters)
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
            partial(load_histogram, parameters=parameters), argsets
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

    total_yield = 0
    templates = []
    for dataset in hist.dataset.unique():
        myhist = hist.loc[hist.dataset == dataset, "hist"].values[0]
        the_hist = myhist[region, channel, "value", :].project(var.name).values()
        the_sumw2 = myhist[region, channel, "sumw2", :].project(var.name).values()
        edges = myhist[region, channel, "value", :].project(var.name).axes[0].edges
        edges = np.array(edges)
        centers = (edges[:-1] + edges[1:]) / 2.0
        total_yield += the_hist.sum()

        name = f"{dataset}_{region}_{channel}"
        th1 = from_numpy([the_hist, edges])
        th1._fName = name
        th1._fSumw2 = np.array(np.append([0], the_sumw2))
        th1._fTsumw2 = np.array(the_sumw2).sum()
        th1._fTsumwx2 = np.array(the_sumw2 * centers).sum()
        templates.append(th1)

    if parameters["save_templates"]:
        path = parameters["templates_path"]
        out_fn = f"{path}/{dataset}_{var.name}_{year}.root"
        save_template(templates, out_fn, parameters)

    return total_yield
