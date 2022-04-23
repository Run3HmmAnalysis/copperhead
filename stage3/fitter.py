import ROOT as rt
import pandas as pd

from python.workflow import parallelize
from python.io import mkdir
from stage2.fit_plots import plot
from stage2.fit_models import chebyshev, doubleCB, bwZ, bwGamma, bwZredux, bernstein

rt.RooMsgService.instance().setGlobalKillBelow(rt.RooFit.ERROR)


def run_fits(client, parameters, df):
    signal_ds = parameters.get("signals", [])
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
    df_fits = pd.DataFrame(columns=["label", "channel", "category", "chi2"])
    for fr in fit_ret:
        df_fits = pd.concat([df_fits, pd.DataFrame.from_dict(fr)])
    # choose fit function with lowest chi2/dof
    df_fits.loc[df_fits.chi2 <= 0, "chi2"] = 999.0
    df_fits.to_pickle("all_chi2.pkl")
    idx = df_fits.groupby(["label", "channel", "category"])["chi2"].idxmin()
    df_fits = (
        df_fits.loc[idx]
        .reset_index()
        .set_index(["label", "channel"])
        .sort_index()
        .drop_duplicates()
    )
    print(df_fits)
    df_fits.to_pickle("best_chi2.pkl")
    return fit_ret


def fitter(args, parameters={}):
    fit_setup = args["fit_setup"]
    df = fit_setup["df"]
    label = fit_setup["label"]
    mode = fit_setup["mode"]
    blinded = fit_setup.get("blinded", False)
    save = parameters.get("save_fits", False)
    save_path = parameters.get("save_fits_path", "fits/")
    channel = args["channel"]
    category = args["category"]

    save_path = save_path + f"/fits_{channel}_{category}/"
    mkdir(save_path)

    df = df[(df.channel == args["channel"]) & (df.category == args["category"])]
    norm = df.lumi_wgt.sum()

    the_fitter = Fitter(
        fitranges={"low": 110, "high": 150, "SR_left": 120, "SR_right": 130},
        fitmodels={
            "bwz": bwZ,
            "bwz_redux": bwZredux,
            "bwgamma": bwGamma,
            "bernstein": bernstein,
            "dcb": doubleCB,
            "chebyshev": chebyshev,
        },
        requires_order=["chebyshev", "bernstein"],
        channel=channel,
        filename_ext="",
    )
    if mode == "bkg":
        chi2 = the_fitter.simple_fit(
            dataset=df,
            label=label,
            category=category,
            blinded=blinded,
            model_names=["bwz", "bwz_redux", "bwgamma"],
            fix_parameters=False,
            store_multipdf=True,
            title="Background",
            save=save,
            save_path=save_path,
            norm=norm,
        )
        # generate and fit pseudo-data
        the_fitter.fit_pseudodata(
            label="pseudodata_" + label,
            category=category,
            blinded=blinded,
            model_names=["bwz", "bwz_redux", "bwgamma"],
            fix_parameters=False,
            title="Pseudo-data",
            save=save,
            save_path=save_path,
            norm=norm,
        )

    if mode == "sig":
        chi2 = the_fitter.simple_fit(
            dataset=df,
            label=label,
            category=category,  # temporary
            blinded=False,
            model_names=["dcb"],
            fix_parameters=True,
            store_multipdf=False,
            title="Signal",
            save=save,
            save_path=save_path,
            norm=norm,
        )
    ret = {"label": label, "channel": channel, "category": category, "chi2": chi2}
    return ret


class Fitter(object):
    def __init__(self, **kwargs):
        self.fitranges = kwargs.get(
            "fitranges", {"low": 110, "high": 150, "SR_left": 120, "SR_right": 130}
        )
        self.fitmodels = kwargs.get("fitmodels", {})
        self.requires_order = kwargs.get("requires_order", [])
        self.channel = kwargs.get("channel", "ggh_0jets")
        self.filename_ext = kwargs.get("filename_ext", "")

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
        orders={},
        fix_parameters=False,
        store_multipdf=False,
        title="",
        save=True,
        save_path="./",
        norm=0,
    ):
        if dataset is None:
            raise Exception("Error: dataset not provided!")
        if len(model_names) == 0:
            raise Exception("Error: empty list of fit models!")

        ds_name = f"ds_{label}"
        self.add_data(dataset, ds_name=ds_name, blinded=blinded)
        ndata = len(dataset["dimuon_mass"].values)

        for model_name in model_names:
            if (model_name in self.requires_order) and (model_name in orders.keys()):
                for order in orders[model_name]:
                    self.add_model(model_name, category=category, order=order)
            else:
                self.add_model(model_name, category=category)

        # self.workspace.Print()
        chi2 = self.fit(
            ds_name,
            ndata,
            model_names,
            orders=orders,
            blinded=blinded,
            fix_parameters=fix_parameters,
            category=category,
            label=label,
            title=title,
            save=save,
            save_path=save_path,
            norm=norm,
        )
        if store_multipdf:
            cat = rt.RooCategory(
                f"pdf_index_{self.channel}_{category}_{label}",
                "index of the active pdf",
            )
            pdflist = rt.RooArgList()
            for model_name in model_names:
                pdflist.add(self.workspace.pdf(model_name))
            multipdf = rt.RooMultiPdf(
                f"multipdf_{self.channel}_{category}_{label}", "multipdf", cat, pdflist
            )
            # self.add_model("multipdf", category=category)
            getattr(self.workspace, "import")(multipdf)
        if save:
            mkdir(save_path)
            self.save_workspace(
                f"{save_path}/workspace_{self.channel}_{category}_{label}{self.filename_ext}"
            )
        return chi2

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
        # w.Import(mass)
        getattr(w, "import")(mass)
        # w.Print()
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
        # self.workspace.Import(data, ds_name)
        getattr(self.workspace, "import")(data, ds_name)

    def fit_pseudodata(
        self,
        label="test",
        category="cat0",
        blinded=False,
        model_names=[],
        orders={},
        fix_parameters=False,
        title="",
        save=True,
        save_path="./",
        norm=0,
    ):
        tag = f"_{self.channel}_{category}"
        chi2 = {}
        model_names_all = []
        for model_name in model_names:
            if (model_name in self.requires_order) and (model_name in orders.keys()):
                for order in orders[model_name]:
                    model_names_all.append({"name": model_name, "order": order})
            else:
                model_names_all.append({"name": model_name, "order": 0})

        for model_names_order in model_names_all:
            model_name = model_names_order["name"]
            order = model_names_order["order"]
            if model_name in self.requires_order:
                model_key = f"{model_name}{order}" + tag
            else:
                model_key = model_name + tag
            # print(model_key)
            # self.workspace.pdf(model_key).Print()
            data = self.workspace.pdf(model_key).generate(
                rt.RooArgSet(self.workspace.obj("mass")), norm
            )
            ds_name = f"pseudodata_{model_key}"
            self.add_data(data, ds_name=ds_name)
            chi2[model_key] = self.fit(
                ds_name,
                norm,
                [model_name],
                orders={model_name: order},
                blinded=blinded,
                fix_parameters=fix_parameters,
                category=category,
                label=label,
                title=title,
                save=save,
                save_path=save_path,
                norm=norm,
            )[model_key]
        if save:
            mkdir(save_path)
            self.save_workspace(
                f"{save_path}/workspace_{self.channel}_{category}_{label}{self.filename_ext}"
            )
        return chi2

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

    def add_model(self, model_name, order=None, category="cat0", prefix=""):
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
        # self.workspace.Import(model)
        getattr(self.workspace, "import")(model)

    def fit(
        self,
        ds_name,
        ndata,
        model_names,
        orders={},
        blinded=False,
        fix_parameters=False,
        save=False,
        save_path="./",
        category="cat0",
        label="",
        title="",
        norm=0,
    ):
        if ds_name not in self.data_registry.keys():
            raise Exception(f"Error: Dataset {ds_name} not in workspace!")

        pdfs = {}
        chi2 = {}
        tag = f"_{self.channel}_{category}"
        model_names_all = []
        for model_name in model_names:
            if (model_name in self.requires_order) and (model_name in orders.keys()):
                for order in orders[model_name]:
                    model_names_all.append(f"{model_name}{order}")
            else:
                model_names_all.append(model_name)
        for model_name in model_names_all:
            model_key = model_name + tag
            pdfs[model_key] = self.workspace.pdf(model_key)
            pdfs[model_key].fitTo(
                self.workspace.obj(ds_name),
                rt.RooFit.Save(),
                rt.RooFit.PrintLevel(-1),
                rt.RooFit.Verbose(rt.kFALSE),
            )
            if fix_parameters:
                pdfs[model_key].getParameters(rt.RooArgSet()).setAttribAll("Constant")
            chi2[model_key] = self.get_chi2(model_key, ds_name, ndata)

            norm_var = rt.RooRealVar(f"{model_key}_norm", f"{model_key}_norm", norm)
            try:
                # self.workspace.Import(norm_var)
                getattr(self.workspace, "import")(norm_var)
            except Exception:
                print(f"{norm_var} already exists in workspace, skipping...")

        if save:
            mkdir(save_path)
            plot(self, ds_name, pdfs, blinded, category, label, title, save_path)

        return chi2

    def get_chi2(self, model_key, ds_name, ndata):
        normalization = rt.RooRealVar(
            "normaliazation", "normalization", ndata, 0.5 * ndata, 2 * ndata
        )
        model = rt.RooExtendPdf(
            "ext", "ext", self.workspace.pdf(model_key), normalization
        )
        xframe = self.workspace.obj("mass").frame()
        ds = self.workspace.obj(ds_name)
        ds.plotOn(xframe, rt.RooFit.Name(ds_name))
        model.plotOn(xframe, rt.RooFit.Name(model_key))
        nparam = model.getParameters(ds).getSize()
        chi2 = xframe.chiSquare(model_key, ds_name, nparam)
        if chi2 <= 0:
            chi2 == 999
        return chi2
