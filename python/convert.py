import pandas as pd
import itertools
from functools import partial
from hist import Hist

from python.variable import Variable
from python.io import save_histogram


def to_histograms(client, parameters, df):
    arglists = {
        "year": df.year.unique(),
        "var_name": parameters["hist_vars"],
        "dataset": df.dataset.unique(),
    }
    argsets = [
        dict(zip(arglists.keys(), values))
        for values in itertools.product(*arglists.values())
    ]

    hist_futures = client.map(
        partial(make_histograms, df=df, parameters=parameters), argsets
    )
    hist_rows = client.gather(hist_futures)
    hist_df = pd.concat(hist_rows).reset_index(drop=True)

    return hist_df


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


def make_histograms(args, df=pd.DataFrame(), parameters={}):
    year = args["year"]
    var_name = args["var_name"]
    dataset = args["dataset"]
    if var_name in parameters["variables_lookup"].keys():
        var = parameters["variables_lookup"][var_name]
    else:
        var = Variable(var_name, var_name, 50, 0, 5)

    regions = parameters["regions"]
    channels = parameters["channels"]

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
    if "score" in var.name:
        bins = parameters["mva_bins"][var.name.replace("score_", "")][f"{year}"]
        hist = hist.Var(bins, name=var.name)
    else:
        hist = hist.Reg(var.nbins, var.xmin, var.xmax, name=var.name, label=var.caption)

    # axis for systematic variation
    if parameters["has_variations"]:
        hist = hist.StrCat(variations, name="variation")

    # container type
    hist = hist.Double()

    for region in regions:
        for w in wgt_variations:
            for v in syst_variations:
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

                for channel in channels:
                    slicer = (
                        (df.dataset == dataset)
                        & (df.region == region)
                        & (df.year == year)
                        & (df[ch_name] == channel)
                    )
                    data = df.loc[slicer, var_name]

                    to_fill = {
                        var.name: data,
                        "region": region,
                        "channel": channel,
                    }
                    to_fill_value = to_fill.copy()
                    to_fill_sumw2 = to_fill.copy()
                    to_fill_value["val_sumw2"] = "value"
                    to_fill_sumw2["val_sumw2"] = "sumw2"

                    if parameters["has_variations"]:
                        to_fill_value["variation"] = variation
                        to_fill_sumw2["variation"] = variation
                        weight = df.loc[slicer, w]
                    else:
                        weight = df.loc[slicer, "lumi_wgt"] * df.loc[slicer, "mc_wgt"]

                    hist.fill(**to_fill_value, weight=weight)
                    hist.fill(**to_fill_sumw2, weight=weight * weight)

    if parameters["save_hists"]:
        save_histogram(hist, var.name, dataset, year, parameters)
    hist_row = pd.DataFrame(
        [{"year": year, "var_name": var.name, "dataset": dataset, "hist": hist}]
    )
    return hist_row
