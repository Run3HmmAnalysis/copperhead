import ROOT as rt
import pandas as pd

from python.workflow import parallelize
from python.io import mkdir
from python.fit_plots import plot
from python.fit_models import chebyshev, doubleCB, bwGamma, bwZredux


def run_fits(client, parameters, df):
    signal_ds = parameters.pop("signals", [])
    all_datasets = df.dataset.unique()
    signals = [ds for ds in all_datasets if ds in signal_ds]
    backgrounds = [ds for ds in all_datasets if ds not in signal_ds]
    fit_setups = []
    if len(backgrounds) > 0:
        fit_setup = {
            "label": "background",
            "mode": "bkg",
            "df": df[df.dataset.isin(backgrounds)],
            "blinded": True,
        }
        fit_setups.append(fit_setup)
    for ds in signals:
        fit_setup = {"label": ds, "mode": "sig", "df": df[df.dataset == ds]}
        fit_setups.append(fit_setup)

    argset = {
        "fit_setup": fit_setups,
        "channel": parameters["mva_channels"],
        "category": df["category"].dropna().unique(),
    }
    fit_ret = parallelize(fitter, argset, client, parameters)
    return fit_ret


def fitter(args, parameters={}):
    fit_setup = args["fit_setup"]
    df = fit_setup["df"]
    label = fit_setup["label"]
    mode = fit_setup["mode"]
    blinded = fit_setup.pop("blinded", False)
    save = parameters.pop("save_fits", False)
    save_path = parameters.pop("save_fits_path", "./")
    channel = args["channel"]
    category = args["category"]

    df = df[(df.channel == args["channel"]) & (df.category == args["category"])]

    print(
        f"Fitter in channel {channel}, category {category}; total nentries = {df.shape[0]}"
    )
    the_fitter = Fitter(
        fitranges={"low": 110, "high": 150, "SR_left": 120, "SR_right": 130},
        fitmodels={
            "bwz_redux": bwZredux,
            "bwgamma": bwGamma,
            "dcb": doubleCB,
            "chebyshev": chebyshev,
        },
        requires_order=["chebyshev"],
        channel=channel,
        filename_ext="",
    )
    if mode == "bkg":
        # background fit should be binned!
        the_fitter.simple_fit(
            dataset=df,
            label=label,
            category=category,  # temporary
            blinded=blinded,
            model_names=["bwz_redux", "bwgamma"],
            fix_parameters=False,
            title="Background",
            save=save,
            save_path=save_path,
        )

    if mode == "sig":
        the_fitter.simple_fit(
            dataset=df,
            label=label,
            category=category,  # temporary
            blinded=False,
            model_names=["dcb"],
            fix_parameters=True,
            title="Signal",
            save=save,
            save_path=save_path,
        )
    return 0


class Fitter(object):
    def __init__(self, **kwargs):
        self.fitranges = kwargs.pop(
            "fitranges", {"low": 110, "high": 150, "SR_left": 120, "SR_right": 130}
        )
        self.fitmodels = kwargs.pop("fitmodels", {})
        self.requires_order = kwargs.pop("requires_order", [])
        self.channel = kwargs.pop("channel", "ggh_0jets")
        self.filename_ext = kwargs.pop("filename_ext", "")

        self.data_registry = {}
        self.model_registry = []

        self.workspace = self.create_workspace()

    def simple_fit(
        self,
        dataset=None,
        label="test",
        category="cat0",
        blinded=False,
        model_names=[],
        fix_parameters=False,
        title="",
        save=True,
        save_path="./",
    ):
        if dataset is None:
            raise Exception("Error: dataset not provided!")
        if len(model_names) == 0:
            raise Exception("Error: empty list of fit models!")

        ds_name = f"ds_{label}"
        self.add_data(dataset, ds_name=ds_name, blinded=blinded)

        for model_name in model_names:
            self.add_model(model_name, category=category)

        self.workspace.Print()
        self.fit(
            ds_name,
            model_names,
            blinded=blinded,
            fix_parameters=fix_parameters,
            category=category,
            label=label,
            title=title,
            save=save,
            save_path=save_path,
        )
        if save:
            mkdir(save_path)
            self.save_workspace(
                f"{save_path}/workspace_{self.channel}_{category}_{label}{self.filename_ext}"
            )

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
        outfile = rt.TFile(f"{out_name}.root", "recreate")
        self.workspace.Write()
        outfile.Close()

    def add_data(self, data, ds_name="ds", blinded=False):
        if ds_name in self.data_registry.keys():
            raise Exception(
                f"Error: Dataset with name {ds_name} already exists in workspace!"
            )

        if isinstance(data, pd.DataFrame):
            data = self.fill_dataset(
                data["dimuon_mass"].values, self.workspace.obj("mass"), ds_name=ds_name
            )
        elif isinstance(data, pd.Series):
            data = self.fill_dataset(
                data.values, self.workspace.obj("mass"), ds_name=ds_name
            )
        elif not (
            isinstance(data, rt.TH1F)
            or isinstance(data, rt.RooDataSet)
            or isinstance(data, rt.RooDataHist)
        ):
            raise Exception(f"Error: trying to add data of wrong type: {type(data)}")

        if blinded:
            data = data.reduce(rt.RooFit.CutRange("sideband_left,sideband_right"))

        self.data_registry[ds_name] = type(data)
        self.workspace.Import(data, ds_name)

    def fill_dataset(self, data, x, ds_name="ds"):
        cols = rt.RooArgSet(x)
        ds = rt.RooDataSet(ds_name, ds_name, cols)
        for datum in data:
            if (datum < x.getMax()) and (datum > x.getMin()):
                x.setVal(datum)
                ds.add(cols)
        return ds

    def generate_data(self, model_name, category, xSec, lumi):
        tag = f"_{self.channel}_{category}"
        model_key = model_name + tag
        if model_key not in self.model_registry:
            self.add_model(model_name, category=category)
        return self.workspace.pdf(model_key).generate(
            rt.RooArgSet(self.workspace.obj("mass")), xSec * lumi
        )

    def add_model(self, model_name, order=None, category="cat0"):
        if model_name not in self.fitmodels.keys():
            raise Exception(f"Error: model {model_name} does not exist!")
        tag = f"_{self.channel}_{category}"
        if order is None:
            model, params = self.fitmodels[model_name](self.workspace.obj("mass"), tag)
        else:
            if model_name in self.requires_order:
                model, params = self.fitmodels[model_name](
                    self.workspace.obj("mass"), tag, order
                )
            else:
                raise Exception(
                    f"Warning: model {model_name} does not require to specify order!"
                )

        model_key = model_name + tag
        if model_key not in self.model_registry:
            self.model_registry.append(model_key)
        self.workspace.Import(model)

    def fit(
        self,
        ds_name,
        model_names,
        blinded=False,
        fix_parameters=False,
        save=False,
        save_path="./",
        category="cat0",
        label="",
        title="",
    ):
        if ds_name not in self.data_registry.keys():
            raise Exception(f"Error: Dataset {ds_name} not in workspace!")

        pdfs = {}
        tag = f"_{self.channel}_{category}"
        for model_name in model_names:
            model_key = model_name + tag
            pdfs[model_key] = self.workspace.pdf(model_key)
            pdfs[model_key].fitTo(self.workspace.obj(ds_name), rt.RooFit.Save())
            if fix_parameters:
                pdfs[model_key].getParameters(rt.RooArgSet()).setAttribAll("Constant")

        if save:
            mkdir(save_path)
            plot(self, ds_name, pdfs, blinded, category, label, title, save_path)
