import itertools
import pandas as pd

from hist import Hist
from python.variable import Variable
from python.io import save_stage2_output_hists


def make_histograms(df, var_name, year, dataset, regions, channels, npart, parameters):
    # try to get binning from config
    if var_name in parameters["variables_lookup"].keys():
        var = parameters["variables_lookup"][var_name]
    else:
        var = Variable(var_name, var_name, 50, 0, 5)

    # prepare list of systematic variations
    wgt_variations = [w for w in df.columns if ("wgt_" in w)]
    syst_variations = parameters.get("syst_variations", ["nominal"])
    variations = []
    for w in wgt_variations:
        for v in syst_variations:
            variation = get_variation(w, v)
            if variation:
                variations.append(variation)

    # prepare multidimensional histogram
    # add axes for (1) mass region, (2) channel, (3) value or sumw2
    hist = (
        Hist.new.StrCat(regions, name="region")
        .StrCat(channels, name="channel")
        .StrCat(["value", "sumw2"], name="val_sumw2")
    )

    # add axis for observable variable
    if ("score" in var.name) and ("mva_bins" in parameters.keys()):
        bins = parameters["mva_bins"][var.name.replace("score_", "")][f"{year}"]
        hist = hist.Var(bins, name=var.name)
    else:
        hist = hist.Reg(var.nbins, var.xmin, var.xmax, name=var.name, label=var.caption)

    # add axis for systematic variation
    hist = hist.StrCat(variations, name="variation")

    # specify container type
    hist = hist.Double()

    # loop over configurations and fill the histogram
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
    hist_info_rows = []
    total_yield = 0
    for loop_arg in loop_args:
        region = loop_arg["region"]
        channel = loop_arg["channel"]
        w = loop_arg["w"]
        v = loop_arg["v"]
        variation = get_variation(w, v)
        if not variation:
            continue

        var_name = f"{var.name}_{v}"
        if var_name not in df.columns:
            if var.name in df.columns:
                var_name = var.name
            else:
                continue

        slicer = (
            (df.dataset == dataset)
            & (df.region == region)
            & (df.year == year)
            & (df[f"channel_{v}"] == channel)
        )
        data = df.loc[slicer, var_name]
        weight = df.loc[slicer, w]

        to_fill = {var.name: data, "region": region, "channel": channel}

        to_fill_value = to_fill.copy()
        to_fill_value["val_sumw2"] = "value"
        to_fill_value["variation"] = variation
        hist.fill(**to_fill_value, weight=weight)

        to_fill_sumw2 = to_fill.copy()
        to_fill_sumw2["val_sumw2"] = "sumw2"
        to_fill_sumw2["variation"] = variation
        hist.fill(**to_fill_sumw2, weight=weight * weight)

        hist_info_row = {
            "year": year,
            "var_name": var.name,
            "dataset": dataset,
            "variation": variation,
            "region": region,
            "channel": channel,
            "yield": weight.sum(),
        }
        if weight.sum() == 0:
            continue
        total_yield += weight.sum()
        if "return_hist" in parameters:
            if parameters["return_hist"]:
                hist_info_row["hist"] = hist
        hist_info_rows.append(hist_info_row)

    if total_yield == 0:
        return None

    # save histogram for this partition to disk
    # (partitions will be joined in stage3)
    save_hists = parameters.get("save_hists", False)
    if save_hists:
        save_stage2_output_hists(hist, var.name, dataset, year, parameters, npart)

    # return info for debugging
    hist_info_rows = pd.DataFrame(hist_info_rows)
    return hist_info_rows


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
