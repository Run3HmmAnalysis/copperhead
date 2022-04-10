import numpy as np
import pandas as pd

from python.workflow import parallelize
from python.variable import Variable
from python.io import (
    load_histogram,
    save_template,
)

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from uproot3_methods.classes.TH1 import from_numpy


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
