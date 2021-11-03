import ROOT as rt


class Fitter(object):
    def __init__(self, **kwargs):
        self.fitranges = kwargs.pop(
            "fitranges", {"low": 110, "high": 150, "SR_left": 120, "SR_right": 130}
        )
        self.workspace = self.create_workspace()
        return

    def create_workspace(self):
        w = rt.RooWorkspace("w", "w")
        mass = rt.RooRealVar(
            "mass", "mass", self.fitranges["low"], self.fitranges["high"]
        )
        mass.setRange(
            "sideband_left", self.fitranges["low"], self.fitranges["SR_left"] + 0.1
        )
        mass.setRange(
            "sideband_right", self.fitranges["SR_right"] - 0.1, self.fitranges["high"]
        )
        mass.setRange("window", self.fitranges["low"], self.fitranges["high"])
        mass.SetTitle("m_{#mu#mu}")
        mass.setUnit("GeV")
        w.Import(mass)
        w.Print()
        return w

    def add_data(self, data, isBlinded, name="", convertData=True):
        if convertData:
            ds = self.filldataset(
                data["dimuon_mass"].values,
                self.workspace.obj("mass"),
                dsName="ds" + name,
            )
            if isBlinded:
                ds = ds.reduce(rt.RooFit.CutRange("sideband_left,sideband_right"))
            self.workspace.Import(ds)
        else:
            if isBlinded:
                data = data.reduce(rt.RooFit.CutRange("sideband_left,sideband_right"))
            self.workspace.Import(data, "ds" + name)

    def filldataset(self, data, x, dsName="ds"):
        cols = rt.RooArgSet(x)
        ds = rt.RooDataSet(dsName, dsName, cols)
        for datum in data:
            if (datum < x.getMax()) and (datum > x.getMin()):
                x.setVal(datum)
                ds.add(cols)
        return ds
