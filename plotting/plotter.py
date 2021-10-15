from functools import partial
import pandas as pd
import numpy as np
from hist.intervals import poisson_interval
from python.utils import load_histograms

import matplotlib.pyplot as plt
import mplhep as hep

style = hep.style.CMS
style["mathtext.fontset"] = "cm"
style["mathtext.default"] = "rm"
plt.style.use(style)


def plotter(client, parameters, hist_df=None, timer=None):
    if hist_df is None:
        # Load histograms
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
    keys = []
    for year in parameters["years"]:
        for var_name in hist_df.var_name.unique():
            if var_name not in parameters["plot_vars"]:
                continue
            hists_to_plot.append(
                hist_df.loc[(hist_df.var_name == var_name) & (hist_df.year == year)]
            )
            keys.append(f"plot: {year} {var_name}")
    if timer:
        timer.add_checkpoint("Loading histograms")

    plot_futures = client.map(
        partial(plot, parameters=parameters), hists_to_plot, key=keys
    )
    yields = client.gather(plot_futures)
    if timer:
        timer.add_checkpoint("Plotting")

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

    stack_groups = parameters["stack_groups"]
    data_groups = parameters["data_groups"]
    step_groups = parameters["step_groups"]

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
                for e, g in parameters["grouping"].items()
                if (g in self.groups) and (e in parameters["datasets"])
            }
            self.entry_list = self.entry_dict.keys()
            self.labels = self.entry_dict.values()
            self.groups = list(set(self.entry_dict.values()))

        def get_plottables(self, hist, year, region, channel, variation, var_name):
            plottables_df = pd.DataFrame(columns=["label", "hist", "sumw2", "integral"])
            all_df = pd.DataFrame(columns=["label", "integral"])
            for group in self.groups:
                group_entries = [e for e, g in self.entry_dict.items() if (group == g)]
                all_labels = hist.loc[
                    hist.dataset.isin(group_entries), "dataset"
                ].values
                try:
                    hist_values_group = [
                        hist[region, channel, variation, "value", :].project(var_name)
                        for hist in hist.loc[
                            hist.dataset.isin(group_entries), "hist"
                        ].values
                    ]
                    hist_sumw2_group = [
                        hist[region, channel, variation, "sumw2", :].project(var_name)
                        for hist in hist.loc[
                            hist.dataset.isin(group_entries), "hist"
                        ].values
                    ]
                except Exception:
                    hist_values_group = [
                        hist[region, channel, "value", :].project(var_name)
                        for hist in hist.loc[
                            hist.dataset.isin(group_entries), "hist"
                        ].values
                    ]
                    hist_sumw2_group = [
                        hist[region, channel, "sumw2", :].project(var_name)
                        for hist in hist.loc[
                            hist.dataset.isin(group_entries), "hist"
                        ].values
                    ]
                if len(hist_values_group) == 0:
                    continue
                nevts = sum(hist_values_group).sum()

                if nevts > 0:
                    plottables_df = plottables_df.append(
                        pd.DataFrame(
                            [
                                {
                                    "label": group,
                                    "hist": sum(hist_values_group),
                                    "sumw2": sum(hist_sumw2_group),
                                    "integral": sum(hist_values_group).sum(),
                                }
                            ]
                        ),
                        ignore_index=True,
                    )
                for label, h in zip(all_labels, hist_values_group):
                    all_df = all_df.append(
                        pd.DataFrame(
                            [
                                {
                                    "label": label,
                                    "integral": h.sum(),
                                }
                            ]
                        ),
                        ignore_index=True,
                    )
            plottables_df.sort_values(by="integral", inplace=True)
            all_df.sort_values(by="integral", inplace=True)
            # print(all_df)
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

    # fig.clf()
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
        plottables_df = entry.get_plottables(
            hist, year, region, channel, variation, var.name
        )
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

    # ax1.set_xlim(var.xmin,var.xmax)
    # ax1.set_xlim(edges[0], edges[-1])
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
