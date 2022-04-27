import numpy as np
import pandas as pd

from python.workflow import parallelize
from python.variable import Variable
from python.io import load_stage2_output_hists, save_template, mkdir

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
    yield_dfs = parallelize(make_templates, argset, client, parameters, seq=True)
    yield_df = pd.concat(yield_dfs).reset_index(drop=True)
    return yield_df


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

    yield_rows = []
    templates = []

    groups = list(set(parameters["grouping"].values()))

    for group in groups:
        datasets = []
        for d in hist_df.dataset.unique():
            if d not in parameters["grouping"].keys():
                continue
            if parameters["grouping"][d] != group:
                continue
            datasets.append(d)

        if len(datasets) == 0:
            continue

        # make a list of systematics;
        # avoid situation where different datasets have incompatible systematics
        wgt_variations = []
        for dataset in datasets:
            n_partitions = len(hist_df.loc[hist_df.dataset == dataset, "hist"].values)
            for i in range(n_partitions):
                new_wgt_vars = list(
                    hist_df.loc[hist_df.dataset == dataset, "hist"]
                    .values[i]
                    .axes["variation"]
                )
                if len(wgt_variations) == 0:
                    wgt_variations = new_wgt_vars
                else:
                    wgt_variations = list(set(wgt_variations) & set(new_wgt_vars))

        for variation in wgt_variations:
            group_hist = []
            group_sumw2 = []

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
            for dataset in datasets:
                try:
                    hist = hist_df.loc[hist_df.dataset == dataset, "hist"].values.sum()
                except Exception:
                    # print(f"Could not merge histograms for {dataset}")
                    continue

                the_hist = hist[slicer_value].project(var.name).values()
                the_sumw2 = hist[slicer_sumw2].project(var.name).values()

                if (the_hist.sum() < 0) or (the_sumw2.sum() < 0):
                    continue

                if len(group_hist) == 0:
                    group_hist = the_hist
                    group_sumw2 = the_sumw2
                else:
                    group_hist += the_hist
                    group_sumw2 += the_sumw2

                edges = hist[slicer_value].project(var.name).axes[0].edges
                edges = np.array(edges)
                centers = (edges[:-1] + edges[1:]) / 2.0

            if len(group_hist) == 0:
                continue

            if group == "Data":
                name = "data_obs"
            else:
                name = group

            if variation == "nominal":
                variation_fixed = variation
            else:
                # TODO: decorrelate LHE, QGL, PDF uncertainties
                variation_fixed = variation.replace("wgt_", "")
                variation_fixed = variation_fixed.replace("_up", "Up")
                variation_fixed = variation_fixed.replace("_down", "Down")
                name = f"{group}_{variation_fixed}"
            th1 = from_numpy([group_hist, edges])
            th1._fName = name
            th1._fSumw2 = np.array(np.append([0], group_sumw2))
            th1._fTsumw2 = np.array(group_sumw2).sum()
            th1._fTsumwx2 = np.array(group_sumw2 * centers).sum()
            templates.append(th1)

            yield_rows.append(
                {
                    "var_name": var_name,
                    "group": group,
                    "region": region,
                    "channel": channel,
                    "year": year,
                    "variation": variation_fixed,
                    "yield": group_hist.sum(),
                }
            )

    if parameters["save_templates"]:
        out_dir = parameters["global_path"]
        mkdir(out_dir)
        out_dir += "/" + parameters["label"]
        mkdir(out_dir)
        out_dir += "/" + "stage3_templates"
        mkdir(out_dir)
        out_dir += "/" + var.name
        mkdir(out_dir)

        out_fn = f"{out_dir}/{channel}_{region}_{year}.root"
        save_template(templates, out_fn, parameters)

    yield_df = pd.DataFrame(yield_rows)
    return yield_df
