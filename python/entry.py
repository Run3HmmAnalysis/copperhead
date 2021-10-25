import pandas as pd


class Entry(object):
    """
    Different types of entries to put on the plot
    """

    def __init__(self, entry_type="step", parameters={}):
        self.entry_type = entry_type
        self.labels = []
        if entry_type == "step":
            self.histtype = "step"
            self.stack = False
            self.plot_opts = {"linewidth": 3}
            self.yerr = False
        elif entry_type == "stack":
            self.histtype = "fill"
            self.stack = True
            self.plot_opts = {"alpha": 0.8, "edgecolor": (0, 0, 0)}
            self.yerr = False
        elif entry_type == "errorbar":
            self.histtype = "errorbar"
            self.stack = False
            self.plot_opts = {"color": "k", "marker": ".", "markersize": 10}
            self.yerr = True
        else:
            raise Exception(f"Wrong entry type: {entry_type}")

        self.groups = parameters["plot_groups"][entry_type]
        self.entry_dict = {
            e: g
            for e, g in parameters["grouping"].items()
            if (g in self.groups) and (e in parameters["datasets"])
        }
        self.entry_list = self.entry_dict.keys()
        self.labels = self.entry_dict.values()
        self.groups = list(set(self.entry_dict.values()))

    def get_plottables(self, hist, year, var_name, slicer):
        slicer[var_name] = slice(None)
        slicer_value = slicer.copy()
        slicer_sumw2 = slicer.copy()
        slicer_value["val_sumw2"] = "value"
        slicer_sumw2["val_sumw2"] = "sumw2"

        plottables_df = pd.DataFrame(columns=["label", "hist", "sumw2", "integral"])

        for group in self.groups:
            group_entries = [e for e, g in self.entry_dict.items() if (group == g)]

            hist_values_group = []
            hist_sumw2_group = []
            for h in hist.loc[hist.dataset.isin(group_entries), "hist"].values:
                hist_values_group.append(h[slicer_value].project(var_name))
                hist_sumw2_group.append(h[slicer_sumw2].project(var_name))

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

        plottables_df.sort_values(by="integral", inplace=True)
        return plottables_df
