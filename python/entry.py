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
