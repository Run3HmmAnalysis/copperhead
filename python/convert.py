import itertools
import numpy as np
import pandas as pd
from hist import Hist
import dask.dataframe as dd

from python.workflow import parallelize
from python.variable import Variable
from python.io import load_histogram, save_histogram, save_template
from python.categorizer import split_into_channels

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from uproot3_methods.classes.TH1 import from_numpy


def to_histograms(client, parameters, df):
    argset = {
        "year": df.year.unique(),
        "var_name": parameters["hist_vars"],
        "dataset": df.dataset.unique(),
    }
    if isinstance(df, pd.DataFrame):
        argset["df"] = df
    elif isinstance(df, dd.DataFrame):
        argset["df"] = [(i, df.partitions[i]) for i in range(df.npartitions)]

    hist_rows = parallelize(make_histograms, argset, client, parameters)
    hist_df = pd.concat(hist_rows).reset_index(drop=True)
    return hist_df


def to_templates(client, parameters, hist_df=None):
    if hist_df is None:
        argset_load = {
            "year": parameters["years"],
            "var_name": parameters["hist_vars"],
            "dataset": parameters["datasets"],
        }
        hist_rows = parallelize(load_histogram, argset_load, client, parameters)
        hist_df = pd.concat(hist_rows).reset_index(drop=True)

    argset = {
        "year": parameters["years"],
        "region": parameters["regions"],
        "channel": parameters["channels"],
        "var_name": [
            v for v in hist_df.var_name.unique() if v in parameters["plot_vars"]
        ],
        "hist_df": [hist_df],
    }
    yields = parallelize(make_templates, argset, client, parameters)
    return yields


def get_variation(wgt_variation, sys_variation):
    if "nominal" in wgt_variation:
        if "nominal" in sys_variation:
            return "nominal"
        else:
            return sys_variation
    else:
        if "nominal" in sys_variation:
            return wgt_variation
        else:
            return None


def make_histograms(args, parameters={}):
    df = args["df"]
    npart = None
    if isinstance(df, tuple):
        npart = df[0]
        df = df[1]

    year = args["year"]
    var_name = args["var_name"]
    dataset = args["dataset"]
    if var_name in parameters["variables_lookup"].keys():
        var = parameters["variables_lookup"][var_name]
    else:
        var = Variable(var_name, var_name, 50, 0, 5)

    regions = parameters["regions"]
    channels = parameters["channels"]

    df = df.compute()
    df.fillna(-999.0, inplace=True)
    split_into_channels(df, v="nominal")

    wgt_variations = ["nominal"]
    syst_variations = ["nominal"]
    variations = []
    if parameters["has_variations"]:
        c_name = "channel nominal"
        wgt_variations = [w for w in df.columns if ("wgt_" in w)]
        syst_variations = parameters["syst_variations"]

        for w in wgt_variations:
            for v in syst_variations:
                variation = get_variation(w, v)
                if variation:
                    variations.append(variation)
    else:
        c_name = "channel"

    regions = [r for r in regions if r in df.region.unique()]
    channels = [c for c in channels if c in df[c_name].unique()]

    # prepare multidimensional histogram
    hist = (
        Hist.new.StrCat(regions, name="region")
        .StrCat(channels, name="channel")
        .StrCat(["value", "sumw2"], name="val_sumw2")
    )

    # axis for observable variable
    if ("score" in var.name) and ("mva_bins" in parameters.keys()):
        bins = parameters["mva_bins"][var.name.replace("score_", "")][f"{year}"]
        hist = hist.Var(bins, name=var.name)
    else:
        hist = hist.Reg(var.nbins, var.xmin, var.xmax, name=var.name, label=var.caption)

    # axis for systematic variation
    if parameters["has_variations"]:
        hist = hist.StrCat(variations, name="variation")

    # container type
    hist = hist.Double()

    loop_args = {
        "region": regions,
        "w": wgt_variations,
        "v": syst_variations,
        "channel": channels,
    }
    loop_args = [
        dict(zip(loop_args.keys(), values))
        for values in itertools.product(*loop_args.values())
    ]
    for loop_arg in loop_args:
        region = loop_arg["region"]
        channel = loop_arg["channel"]
        w = loop_arg["w"]
        v = loop_arg["v"]
        variation = get_variation(w, v)
        if not variation:
            continue
        if parameters["has_variations"]:
            var_name = f"{var.name} {v}"
            ch_name = f"channel {v}"
            if var_name not in df.columns:
                if var.name in df.columns:
                    var_name = var.name
                else:
                    continue
        else:
            var_name = var.name
            ch_name = "channel"

        slicer = (
            (df.dataset == dataset)
            & (df.region == region)
            & (df.year == year)
            & (df[ch_name] == channel)
        )
        data = df.loc[slicer, var_name]

        to_fill = {var.name: data, "region": region, "channel": channel}
        to_fill_value = to_fill.copy()
        to_fill_sumw2 = to_fill.copy()
        to_fill_value["val_sumw2"] = "value"
        to_fill_sumw2["val_sumw2"] = "sumw2"

        if parameters["has_variations"]:
            to_fill_value["variation"] = variation
            to_fill_sumw2["variation"] = variation
            weight = df.loc[slicer, w]
        else:
            weight = df.loc[slicer, "lumi_wgt"]  # * df.loc[slicer, "mc_wgt"]

        hist.fill(**to_fill_value, weight=weight)
        hist.fill(**to_fill_sumw2, weight=weight * weight)

    if parameters["save_hists"]:
        save_histogram(hist, var.name, dataset, year, parameters, npart)
    hist_row = pd.DataFrame(
        [{"year": year, "var_name": var.name, "dataset": dataset, "hist": hist}]
    )
    return hist_row


def make_templates(args, parameters={}):
    year = args["year"]
    region = args["region"]
    channel = args["channel"]
    var_name = args["var_name"]
    hist = args["hist_df"].loc[
        (args["hist_df"].var_name == var_name) & (args["hist_df"].year == year)
    ]

    if var_name in parameters["variables_lookup"].keys():
        var = parameters["variables_lookup"][var_name]
    else:
        var = Variable(var_name, var_name, 50, 0, 5)

    if hist.shape[0] == 0:
        return

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
