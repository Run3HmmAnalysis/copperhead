import os
from functools import partial

import dask.dataframe as dd
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist
from hist.intervals import poisson_interval
from config.variables import variables_lookup, Variable
from python.utils import mkdir

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
pd.options.mode.chained_assignment = None

grouping = {
    # "ggh_powheg": "ggH",
    # "vbf_powheg": "VBF",
    "dy_m100_mg": "DY",
    "ttbar_dl": "TTbar",
    "tttj": "TTbar",
    "tttt": "TTbar",
    "tttw": "TTbar",
    "ttwj": "TTbar",
    "ttww": "TTbar",
    "ttz": "TTbar",
    "st_s": "Single top",
    # "st_t_top": "Single top",
    "st_t_antitop": "Single top",
    "st_tw_top": "Single top",
    "st_tw_antitop": "Single top",
    "zz_2l2q": "VV",
}

grouping_alt = {
    "DY": ["dy_m100_mg"],
    # "EWK": [],
    "TTbar": ["ttbar_dl", "tttj", "tttt", "tttw", "ttwj", "ttww", "ttz"],
    "Single top": ["st_s", "st_t_antitop", "st_tw_top", "st_tw_antitop"],
    "VV": ["zz_2l2q"],
    # "VVV": [],
    # "ggH": ["ggh_amcPS"],
    # "VBF": ["vbf_powheg_dipole"],
}


def workflow(client, paths, parameters, timer):
    # Load dataframes
    df_future = client.map(load_data, paths)
    df_future = client.gather(df_future)
    timer.add_checkpoint("Loaded data from Parquet")

    # Merge dataframes
    try:
        df = dd.concat([d for d in df_future if len(d.columns) > 0])
    except Exception:
        return
    npart = df.npartitions
    df = df.compute()
    df.reset_index(inplace=True, drop=True)
    df = dd.from_pandas(df, npartitions=npart)
    if npart > 2 * parameters["ncpus"]:
        df = df.repartition(npartitions=parameters["ncpus"])
    timer.add_checkpoint("Combined into a single Dask DataFrame")

    keep_columns = ["s", "year", "r", "c"]
    keep_columns += [c for c in df.columns if "wgt" in c]
    keep_columns += parameters["hist_vars"]

    df = df[[c for c in keep_columns if c in df.columns]]
    df = df.compute()

    df.dropna(axis=1, inplace=True)
    df.reset_index(inplace=True)
    timer.add_checkpoint("Prepared for histogramming")

    argsets = []
    for year in df.year.unique():
        for var_name in parameters["hist_vars"]:
            for dataset in df.s.unique():
                argsets.append({"year": year, "var_name": var_name, "dataset": dataset})
    # Make histograms
    hist_futures = client.map(partial(histogram, df=df, parameters=parameters), argsets)
    client.gather(hist_futures)
    # hist_rows = client.gather(hist_futures)
    # hist_df = pd.concat(hist_rows).reset_index(drop=True)
    timer.add_checkpoint("Histogramming")


def plotter(client, parameters, timer):
    # Load histograms
    argsets = []
    for year in parameters["years"]:
        for var_name in parameters["hist_vars"]:
            for dataset in parameters["datasets"]:
                argsets.append({"year": year, "var_name": var_name, "dataset": dataset})
    hist_futures = client.map(partial(load_histograms, parameters=parameters), argsets)
    hist_rows = client.gather(hist_futures)
    hist_df = pd.concat(hist_rows).reset_index(drop=True)

    hists_to_plot = []
    keys = []
    for year in parameters["years"]:
        for var_name in hist_df.var_name.unique():
            if var_name not in parameters["plot_vars"]:
                continue
            hists_to_plot.append(
                hist_df.loc[(hist_df.var_name == var_name) & (hist_df.year == year)]
            )
            keys.append(f"plot: {year} {var_name}")

    timer.add_checkpoint("Loading histograms")

    plot_futures = client.map(
        partial(plot, parameters=parameters), hists_to_plot, key=keys
    )
    client.gather(plot_futures)

    timer.add_checkpoint("Plotting")


def load_data(path):
    df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    if len(path) > 0:
        try:
            df = dd.read_parquet(path)
        except Exception:
            return df
    df["c"] = "vbf"
    return df


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

    regions = [r for r in regions if r in df.r.unique()]
    channels = [c for c in channels if c in df["c"].unique()]

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

    for r in regions:
        var_name = var.name
        if var.name in df.columns:
            var_name = var.name
        else:
            continue
        for c in channels:
            slicer = (df.s == dataset) & (df.r == r) & (df.year == year) & (df.c == c)
            data = df.loc[slicer, var_name]
            weight = df.loc[slicer, "lumi_wgt"] * df.loc[slicer, "mc_wgt"]
            print(dataset, df.loc[slicer, "lumi_wgt"].mean())
            hist.fill(r, c, "value", data, weight=weight)
            hist.fill(r, c, "sumw2", data, weight=weight * weight)
            # TODO: add treatment of PDF systematics
            # (MC replicas)
    save_hist(hist, var.name, dataset, year, parameters)
    hist_row = pd.DataFrame(
        [{"year": year, "var_name": var.name, "dataset": dataset, "hist": hist}]
    )
    return hist_row


def save_hist(hist, var_name, dataset, year, parameters):
    mkdir(parameters["hist_path"])
    hist_path = parameters["hist_path"] + parameters["label"]
    mkdir(hist_path)
    mkdir(f"{hist_path}/{year}")
    mkdir(f"{hist_path}/{year}/{var_name}")
    path = f"{hist_path}/{year}/{var_name}/{dataset}.pickle"
    with open(path, "wb") as handle:
        pickle.dump(hist, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_histograms(argset, parameters):
    year = argset["year"]
    var_name = argset["var_name"]
    dataset = argset["dataset"]
    hist_path = parameters["hist_path"] + parameters["label"]
    path = f"{hist_path}/{year}/{var_name}/{dataset}.pickle"
    try:
        with open(path, "rb") as handle:
            hist = pickle.load(handle)
    except Exception:
        return pd.DataFrame()
    hist_row = pd.DataFrame(
        [{"year": year, "var_name": var_name, "dataset": dataset, "hist": hist}]
    )
    return hist_row


def plot(hist, parameters={}):
    if hist.shape[0] == 0:
        return
    var = hist["hist"].values[0].axes[-1]
    plotsize = 8
    ratio_plot_size = 0.25

    # temporary
    r = "h-peak"
    c = "vbf"
    years = hist.year.unique()
    if len(years) > 1:
        print(
            f"Histograms for more than one year provided. Will make plots only for {years[0]}."
        )
    year = years[0]

    stack_groups = ["DY", "EWK", "TTbar", "Single top", "VV", "VVV"]
    data_groups = ["Data"]
    step_groups = ["VBF", "ggH"]

    class Entry(object):
        """
        Different types of entries to put on the plot
        """

        def __init__(self, entry_type="step", parameters={}):
            self.entry_type = entry_type
            self.labels = []
            if entry_type == "step":
                self.groups = step_groups
                self.histtype = "step"
                self.stack = False
                self.plot_opts = {"linewidth": 3}
                self.yerr = False
            elif entry_type == "stack":
                self.groups = stack_groups
                self.histtype = "fill"
                self.stack = True
                self.plot_opts = {"alpha": 0.8, "edgecolor": (0, 0, 0)}
                self.yerr = False
            elif entry_type == "data":
                self.groups = data_groups
                self.histtype = "errorbar"
                self.stack = False
                self.plot_opts = {"color": "k", "marker": ".", "markersize": 10}
                self.yerr = True
            else:
                raise Exception(f"Wrong entry type: {entry_type}")

            self.entry_dict = {
                e: g
                for e, g in grouping.items()
                if (g in self.groups) and (e in parameters["datasets"])
            }
            self.entry_list = self.entry_dict.keys()
            self.labels = self.entry_dict.values()
            self.groups = list(set(self.entry_dict.values()))

        def get_plottables(self, hist, year, r, c, var_name):
            plottables_df = pd.DataFrame(columns=["label", "hist", "sumw2", "integral"])
            all_df = pd.DataFrame(columns=["label", "integral"])
            for group in self.groups:
                group_entries = [e for e, g in self.entry_dict.items() if (group == g)]
                all_labels = hist.loc[
                    hist.dataset.isin(group_entries), "dataset"
                ].values
                hist_values_group = [
                    hist[r, c, "value", :].project(var_name)
                    for hist in hist.loc[
                        hist.dataset.isin(group_entries), "hist"
                    ].values
                ]
                hist_sumw2_group = [
                    hist[r, c, "sumw2", :].project(var_name)
                    for hist in hist.loc[
                        hist.dataset.isin(group_entries), "hist"
                    ].values
                ]
                if len(hist_values_group) == 0:
                    continue
                nevts = sum(hist_values_group).sum()

                if nevts > 0:
                    plottables_df = plottables_df.append(
                        {
                            "label": group,
                            "hist": sum(hist_values_group),
                            "sumw2": sum(hist_sumw2_group),
                            "integral": sum(hist_values_group).sum(),
                        },
                        ignore_index=True,
                    )
                for label, h in zip(all_labels, hist_values_group):
                    all_df = all_df.append(
                        {
                            "label": label,
                            "integral": h.sum(),
                        },
                        ignore_index=True,
                    )
            plottables_df.sort_values(by="integral", inplace=True)
            all_df.sort_values(by="integral", inplace=True)
            print(all_df)
            return plottables_df

    stat_err_opts = {
        "step": "post",
        "label": "Stat. unc.",
        "hatch": "//////",
        "facecolor": "none",
        "edgecolor": (0, 0, 0, 0.5),
        "linewidth": 0,
    }
    ratio_err_opts = {"step": "post", "facecolor": (0, 0, 0, 0.3), "linewidth": 0}

    entry_types = ["stack", "step", "data"]
    entries = {et: Entry(et, parameters) for et in entry_types}

    fig = plt.figure()
    style = hep.style.CMS
    style["mathtext.fontset"] = "cm"
    style["mathtext.default"] = "rm"
    plt.style.use(style)

    fig.clf()
    fig.set_size_inches(plotsize * 1.2, plotsize * (1 + ratio_plot_size))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[(1 - ratio_plot_size), ratio_plot_size], hspace=0.07
    )

    # Top panel: Data/MC
    plt1 = fig.add_subplot(gs[0])

    for entry in entries.values():
        if len(entry.entry_list) == 0:
            continue
        plottables_df = entry.get_plottables(hist, year, r, c, var.name)
        plottables = plottables_df["hist"].values.tolist()
        sumw2 = plottables_df["sumw2"].values.tolist()
        labels = plottables_df["label"].values.tolist()

        if len(plottables) == 0:
            continue

        yerr = np.sqrt(sum(plottables).values()) if entry.yerr else None

        hep.histplot(
            plottables,
            label=labels,
            ax=plt1,
            yerr=yerr,
            stack=entry.stack,
            histtype=entry.histtype,
            **entry.plot_opts,
        )

        # MC errors
        if entry.entry_type == "stack":
            total_bkg = sum(plottables).values()
            total_sumw2 = sum(sumw2).values()
            if sum(total_bkg) > 0:
                err = poisson_interval(total_bkg, total_sumw2)
                plt1.fill_between(
                    x=plottables[0].axes[0].edges,
                    y1=np.r_[err[0, :], err[0, -1]],
                    y2=np.r_[err[1, :], err[1, -1]],
                    **stat_err_opts,
                )

    plt1.set_yscale("log")
    plt1.set_ylim(0.01, 1e9)
    # plt1.set_xlim(var.xmin,var.xmax)
    # plt1.set_xlim(edges[0], edges[-1])
    plt1.set_xlabel("")
    plt1.tick_params(axis="x", labelbottom=False)
    plt1.legend(prop={"size": "x-small"})

    # Bottom panel: Data/MC ratio plot
    plt2 = fig.add_subplot(gs[1], sharex=plt1)

    num = den = []

    if len(entries["data"].entry_list) > 0:
        # get Data yields
        num_df = entries["data"].get_plottables(hist, year, r, c, var.name)
        num = num_df["hist"].values.tolist()
        if len(num) > 0:
            num = sum(num).values()

    if len(entries["stack"].entry_list) > 0:
        # get MC yields and sumw2
        den_df = entries["stack"].get_plottables(hist, year, r, c, var.name)
        den = den_df["hist"].values.tolist()
        den_sumw2 = den_df["sumw2"].values.tolist()
        if len(den) > 0:
            edges = den[0].axes[0].edges
            den = sum(den).values()  # total MC
            den_sumw2 = sum(den_sumw2).values()

    if len(num) * len(den) > 0:
        # compute Data/MC ratio
        ratio = np.divide(num, den)
        yerr = np.zeros_like(num)
        yerr[den > 0] = np.sqrt(num[den > 0]) / den[den > 0]
        hep.histplot(
            ratio,
            bins=edges,
            ax=plt2,
            yerr=yerr,
            histtype="errorbar",
            **entries["data"].plot_opts,
        )

    if sum(den) > 0:
        # compute MC uncertainty
        unity = np.ones_like(den)
        w2 = np.zeros_like(den)
        w2[den > 0] = den_sumw2[den > 0] / den[den > 0] ** 2
        den_unc = poisson_interval(unity, w2)
        plt2.fill_between(
            edges,
            np.r_[den_unc[0], den_unc[0, -1]],
            np.r_[den_unc[1], den_unc[1, -1]],
            label="Stat. unc.",
            **ratio_err_opts,
        )

    plt2.axhline(1, ls="--")
    plt2.set_ylim([0.5, 1.5])
    plt2.set_ylabel("Data/MC", loc="center")
    plt2.set_xlabel(var.label, loc="right")
    plt2.legend(prop={"size": "x-small"})

    hep.cms.label(
        ax=plt1, data=False, label="Preliminary", year="HL-LHC", rlabel="14 TeV"
    )

    path = parameters["plots_path"]
    out_name = f"{path}/{var.name}_{year}.png"
    fig.savefig(out_name)
    print(f"Saved: {out_name}")
    return
