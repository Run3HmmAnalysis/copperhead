import numpy as np
import pandas as pd

from python.workflow import parallelize
from python.variable import Variable
from python.io import (
    load_stage2_output_hists,
    save_template,
)

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from uproot3_methods.classes.TH1 import from_numpy


def to_templates(client, parameters, hist_df=None):
    if hist_df is None:
        argset_load = {
            "year": parameters["years"],
            "var_name": parameters["templates_vars"],
            "dataset": parameters["datasets"],
        }
        hist_rows = parallelize(
            load_stage2_output_hists, argset_load, client, parameters
        )
        hist_df = pd.concat(hist_rows).reset_index(drop=True)
        if hist_df.shape[0] == 0:
            print("No templates to create!")
            return []

    argset = {
        "year": parameters["years"],
        "region": parameters["regions"],
        "channel": parameters["channels"],
        "var_name": [
            v for v in hist_df.var_name.unique() if v in parameters["templates_vars"]
        ],
        "hist_df": [hist_df],
    }

    yields = parallelize(make_templates, argset, client, parameters)
    return yields


def make_templates(args, parameters={}):
    year = args["year"]
    region = args["region"]
    channel = args["channel"]
    var_name = args["var_name"]
    hist_df = args["hist_df"].loc[
        (args["hist_df"].var_name == var_name) & (args["hist_df"].year == year)
    ]

    if var_name in parameters["variables_lookup"].keys():
        var = parameters["variables_lookup"][var_name]
    else:
        var = Variable(var_name, var_name, 50, 0, 5)

    if hist_df.shape[0] == 0:
        return

    variation = "nominal"
    slicer_value = {
        "region": region,
        "channel": channel,
        "variation": variation,
        "val_sumw2": "value",
    }
    slicer_sumw2 = {
        "region": region,
        "channel": channel,
        "variation": variation,
        "val_sumw2": "sumw2",
    }

    total_yield = 0
    templates = []

    # TODO: add loop for systematics

    groups = list(set(parameters["grouping"].values()))

    for group in groups:
        group_hist = []
        group_sumw2 = []
        datasets = []
        for d in hist_df.dataset.unique():
            if d not in parameters["grouping"].keys():
                continue
            if parameters["grouping"][d] != group:
                continue
            datasets.append(d)

        if len(datasets) == 0:
            continue

        for dataset in datasets:
            try:
                hist = hist_df.loc[hist_df.dataset == dataset, "hist"].values.sum()
            except Exception:
                print(f"Could not merge histograms for {dataset}")
                continue

            the_hist = hist[slicer_value].project(var.name).values()
            the_sumw2 = hist[slicer_sumw2].project(var.name).values()
            if len(group_hist) == 0:
                group_hist = the_hist
                group_sumw2 = the_sumw2
            else:
                group_hist += the_hist
                group_sumw2 += the_sumw2

            edges = hist[slicer_value].project(var.name).axes[0].edges
            edges = np.array(edges)
            centers = (edges[:-1] + edges[1:]) / 2.0
            total_yield += the_hist.sum()

        th1 = from_numpy([group_hist, edges])
        th1._fName = group
        th1._fSumw2 = np.array(np.append([0], group_sumw2))
        th1._fTsumw2 = np.array(group_sumw2).sum()
        th1._fTsumwx2 = np.array(group_sumw2 * centers).sum()
        templates.append(th1)

    if parameters["save_templates"]:
        path = parameters["templates_path"]
        out_fn = f"{path}/{var.name}_{region}_{channel}_{year}.root"
        save_template(templates, out_fn, parameters)

    return total_yield
