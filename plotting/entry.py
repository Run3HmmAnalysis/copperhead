import pandas as pd


class Entry(object):
    """
    Different types of entries to put on the plot
    """

    def __init__(self, entry_type="step", parameters={}):
        self.entry_type = entry_type
        self.labels = []
        if entry_type == "step":
            self.groups = parameters["plot_groups"]["step"]
            self.histtype = "step"
            self.stack = False
            self.plot_opts = {"linewidth": 3}
            self.yerr = False
        elif entry_type == "stack":
            self.groups = parameters["plot_groups"]["stack"]
            self.histtype = "fill"
            self.stack = True
            self.plot_opts = {"alpha": 0.8, "edgecolor": (0, 0, 0)}
            self.yerr = False
        elif entry_type == "data":
            self.groups = parameters["plot_groups"]["data"]
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

    def get_plottables(self, hist, year, var_name, dimensions):
        slicer_value = tuple(list(dimensions) + ["value", slice(None)])
        slicer_sumw2 = tuple(list(dimensions) + ["sumw2", slice(None)])
        plottables_df = pd.DataFrame(columns=["label", "hist", "sumw2", "integral"])
        all_df = pd.DataFrame(columns=["label", "integral"])
        for group in self.groups:
            group_entries = [e for e, g in self.entry_dict.items() if (group == g)]
            all_labels = hist.loc[hist.dataset.isin(group_entries), "dataset"].values
            hist_values_group = [
                hist[slicer_value].project(var_name)
                for hist in hist.loc[hist.dataset.isin(group_entries), "hist"].values
            ]
            hist_sumw2_group = [
                hist[slicer_sumw2].project(var_name)
                for hist in hist.loc[hist.dataset.isin(group_entries), "hist"].values
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
