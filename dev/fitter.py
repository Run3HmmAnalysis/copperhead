import ROOT as rt
import pandas as pd
from fit_plots import plot


class Fitter(object):
    def __init__(self, **kwargs):
        self.fitranges = kwargs.pop(
            "fitranges", {"low": 110, "high": 150, "SR_left": 120, "SR_right": 130}
        )
        self.fitmodels = kwargs.pop("fitmodels", {})
        self.requires_order = kwargs.pop("requires_order", [])
        self.process = kwargs.pop("process", "ggH")
        self.category = kwargs.pop("category", "cat0")
        self.tag = f"_{self.process}_{self.category}"

        self.data_registry = {}
        self.model_registry = []

        self.workspace = self.create_workspace()

    def simple_fit(
        self,
        dataset=None,
        ds_name="",
        blinded=False,
        models=[],
        fix_parameters=False,
        label="",
        title="",
        save=True,
        ws_out_name="",
    ):
        if (dataset is None) or (ds_name == "") or (len(models) == 0):
            return
        self.add_data(dataset, name=ds_name, blinded=blinded)
        self.add_models(models)
        self.workspace.Print()
        self.fit(
            ds_name,
            models,
            blinded=blinded,
            fix_parameters=fix_parameters,
            label=label,
            title=title,
            save=save,
        )
        if save:
            self.save_workspace(ws_out_name)

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

    def save_workspace(self, out_name):
        outfile = rt.TFile(out_name + ".root", "recreate")
        self.workspace.Write()
        outfile.Close()
        del self.workspace

    def add_data(self, data, name="ds", blinded=False):
        if name in self.data_registry.keys():
            raise Exception(
                f"Error: Dataset with name {name} already exists in workspace!"
            )

        if isinstance(data, pd.DataFrame):
            data = self.fill_dataset(
                data["dimuon_mass"].values, self.workspace.obj("mass"), name=name
            )
        elif not (
            isinstance(data, rt.TH1F)
            or isinstance(data, rt.RooDataSet)
            or isinstance(data, rt.RooDataHist)
        ):
            raise Exception(f"Error: trying to add data of wrong type: {type(data)}")

        if blinded:
            data = data.reduce(rt.RooFit.CutRange("sideband_left,sideband_right"))

        self.data_registry[name] = type(data)
        self.workspace.Import(data, name)

    def fill_dataset(self, data, x, name="ds"):
        cols = rt.RooArgSet(x)
        ds = rt.RooDataSet(name, name, cols)
        for datum in data:
            if (datum < x.getMax()) and (datum > x.getMin()):
                x.setVal(datum)
                ds.add(cols)
        return ds

    def generate_data(self, pdf_name, tag, xSec, lumi):
        if pdf_name not in self.model_registry:
            self.add_model(pdf_name)
        return self.workspace.pdf(pdf_name + tag).generate(
            rt.RooArgSet(self.workspace.obj("mass")), xSec * lumi
        )

    def add_models(self, models):
        for model in models:
            if isinstance(model, dict):
                if "name" not in model.keys():
                    continue
                order = model.pop("order", None)
                tag = model.pop("tag", None)
                self.add_model(model["name"], order=order, tag=tag)
            elif isinstance(model, str):
                self.add_model(model)

    def add_model(self, model_name, order=None, tag=None):
        if model_name not in self.fitmodels.keys():
            print(f"Error: model {model_name} does not exist!")
            return

        if tag is None:
            tag = self.tag

        if order is None:
            model, params = self.fitmodels[model_name](self.workspace.obj("mass"), tag)
        else:
            if model_name in self.requires_order:
                model, params = self.fitmodels[model_name](
                    self.workspace.obj("mass"), tag, order
                )
            else:
                print(f"Warning: model {model_name} does not require to specify order!")
                model, params = self.fitmodels[model_name](
                    self.workspace.obj("mass"), tag
                )
        if model_name not in self.model_registry:
            self.model_registry.append(model_name)
        self.workspace.Import(model)

    def fit(
        self,
        ds_name,
        model_names,
        blinded=False,
        fix_parameters=False,
        tag=None,
        save=False,
        label="",
        title="",
    ):
        if ds_name not in self.data_registry.keys():
            raise Exception(f"Error: Dataset {ds_name} not in workspace!")

        print(f"In cat {self.category}")
        pdfs = {}
        if tag is None:
            tag = self.tag
        for model in model_names:
            pdfs[model + tag] = self.workspace.pdf(model + tag)
            pdfs[model + tag].fitTo(self.workspace.obj(ds_name), rt.RooFit.Save())
            if fix_parameters:
                pdfs[model + tag].getParameters(rt.RooArgSet()).setAttribAll("Constant")

        if save:
            plot(
                self.workspace,
                ds_name,
                pdfs,
                blinded,
                tag.split("_")[1],
                tag.split("_")[2],
                label,
                title,
            )
