from functools import partial
import pandas as pd
import numpy as np
from hist.intervals import poisson_interval
from python.utils import load_histograms
from plotting.entry import Entry

import matplotlib.pyplot as plt
import mplhep as hep

style = hep.style.CMS
style["mathtext.fontset"] = "cm"
style["mathtext.default"] = "rm"
plt.style.use(style)

stat_err_opts = {
    "step": "post",
    "label": "Stat. unc.",
    "hatch": "//////",
    "facecolor": "none",
    "edgecolor": (0, 0, 0, 0.5),
    "linewidth": 0,
}
ratio_err_opts = {"step": "post", "facecolor": (0, 0, 0, 0.3), "linewidth": 0}


def plotter(client, parameters, hist_df=None, timer=None):

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

    hists_to_plot = []
    for year in parameters["years"]:
        for var_name in hist_df.var_name.unique():
            if var_name not in parameters["plot_vars"]:
                continue
            hists_to_plot.append(
                hist_df.loc[(hist_df.var_name == var_name) & (hist_df.year == year)]
            )

    plot_futures = client.map(partial(plot, parameters=parameters), hists_to_plot)
    yields = client.gather(plot_futures)

    return yields


def plot(hist, parameters={}):
    if hist.shape[0] == 0:
        return
    var = hist["hist"].values[0].axes[-1]
    plotsize = 8
    ratio_plot_size = 0.25

    # temporary
    region = "h-peak"
    channel = "vbf"
    variation = "nominal"
    years = hist.year.unique()
    if len(years) > 1:
        print(
            f"Histograms for more than one year provided. Will make plots only for {years[0]}."
        )
    year = years[0]

    entries = {et: Entry(et, parameters) for et in parameters["plot_groups"].keys()}

    fig = plt.figure()

    if parameters["plot_ratio"]:
        fig.set_size_inches(plotsize * 1.2, plotsize * (1 + ratio_plot_size))
        gs = fig.add_gridspec(
            2, 1, height_ratios=[(1 - ratio_plot_size), ratio_plot_size], hspace=0.07
        )
        # Top panel: Data/MC
        ax1 = fig.add_subplot(gs[0])
    else:
        fig, ax1 = plt.subplots()
        fig.set_size_inches(plotsize, plotsize)

    total_yield = 0
    for entry in entries.values():
        if len(entry.entry_list) == 0:
            continue
        if parameters["has_variations"]:
            dimensions = (region, channel, variation)
        else:
            dimensions = (region, channel)
        plottables_df = entry.get_plottables(hist, year, var.name, dimensions)
        plottables = plottables_df["hist"].values.tolist()
        sumw2 = plottables_df["sumw2"].values.tolist()
        labels = plottables_df["label"].values.tolist()
        total_yield += sum([p.sum() for p in plottables])

        if len(plottables) == 0:
            continue

        yerr = np.sqrt(sum(plottables).values()) if entry.yerr else None

        hep.histplot(
            plottables,
            label=labels,
            ax=ax1,
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
                ax1.fill_between(
                    x=plottables[0].axes[0].edges,
                    y1=np.r_[err[0, :], err[0, -1]],
                    y2=np.r_[err[1, :], err[1, -1]],
                    **stat_err_opts,
                )

    ax1.set_yscale("log")
    ax1.set_ylim(0.01, 1e9)
    ax1.legend(prop={"size": "x-small"})

    if parameters["plot_ratio"]:
        ax1.set_xlabel("")
        ax1.tick_params(axis="x", labelbottom=False)
    else:
        ax1.set_xlabel(var.label, loc="right")

    if parameters["plot_ratio"]:

        # Bottom panel: Data/MC ratio plot
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        num = den = []

        if len(entries["data"].entry_list) > 0:
            # get Data yields
            num_df = entries["data"].get_plottables(
                hist, year, region, channel, variation, var.name
            )
            num = num_df["hist"].values.tolist()
            if len(num) > 0:
                num = sum(num).values()

        if len(entries["stack"].entry_list) > 0:
            # get MC yields and sumw2
            den_df = entries["stack"].get_plottables(
                hist, year, region, channel, variation, var.name
            )
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
                ax=ax2,
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
            ax2.fill_between(
                edges,
                np.r_[den_unc[0], den_unc[0, -1]],
                np.r_[den_unc[1], den_unc[1, -1]],
                label="Stat. unc.",
                **ratio_err_opts,
            )

        ax2.axhline(1, ls="--")
        ax2.set_ylim([0.5, 1.5])
        ax2.set_ylabel("Data/MC", loc="center")
        ax2.set_xlabel(var.label, loc="right")
        ax2.legend(prop={"size": "x-small"})

    if parameters["14TeV_label"]:
        hep.cms.label(
            ax=ax1, data=False, label="Preliminary", year="HL-LHC", rlabel="14 TeV"
        )
    else:
        hep.cms.label(ax=ax1, data=True, label="Preliminary", year=year)

    if parameters["save_plots"]:
        path = parameters["plots_path"]
        out_name = f"{path}/{var.name}_{year}.png"
        fig.savefig(out_name)
        print(f"Saved: {out_name}")

    return total_yield
