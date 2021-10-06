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


training_features = [
    "dimuon_mass",
    "dimuon_pt",
    "dimuon_pt_log",
    "dimuon_eta",
    "dimuon_mass_res",
    "dimuon_mass_res_rel",
    "dimuon_cos_theta_cs",
    "dimuon_phi_cs",
    "jet1_pt",
    "jet1_eta",
    "jet1_phi",
    "jet1_qgl",
    "jet2_pt",
    "jet2_eta",
    "jet2_phi",
    "jet2_qgl",
    "jj_mass",
    "jj_mass_log",
    "jj_dEta",
    "rpt",
    "ll_zstar_log",
    "mmj_min_dEta",
    "nsoftjets5",
    "htsoft2",
]

training_features = [
    "dimuon_mass",
    "dimuon_pt",
    "dimuon_pt_log",
    "dimuon_eta",
    "dimuon_ebe_mass_res",
    "dimuon_ebe_mass_res_rel",
    "dimuon_cos_theta_cs",
    "dimuon_phi_cs",
    "jet1_pt",
    "jet1_eta",
    "jet1_phi",
    "jet1_qgl",
    "jet2_pt",
    "jet2_eta",
    "jet2_phi",
    "jet2_qgl",
    "jj_mass",
    "jj_mass_log",
    "jj_dEta",
    "rpt",
    "ll_zstar_log",
    "mmj_min_dEta",
    "nsoftjets5",
    "htsoft2",
]

grouping = {
    "data_A": "Data",
    "data_B": "Data",
    "data_C": "Data",
    "data_D": "Data",
    "data_E": "Data",
    "data_F": "Data",
    "data_G": "Data",
    "data_H": "Data",
    # 'dy_0j': 'DY',
    # 'dy_1j': 'DY',
    # 'dy_2j': 'DY',
    # 'dy_m105_160_amc': 'DY_nofilter',
    # 'dy_m105_160_vbf_amc': 'DY_filter',
    "dy_m105_160_amc": "DY",
    "dy_m105_160_vbf_amc": "DY",
    "ewk_lljj_mll105_160_ptj0": "EWK",
    # 'ewk_lljj_mll105_160_py_dipole': 'EWK_Pythia',
    "ttjets_dl": "TT+ST",
    "ttjets_sl": "TT+ST",
    "ttw": "TT+ST",
    "ttz": "TT+ST",
    "st_tw_top": "TT+ST",
    "st_tw_antitop": "TT+ST",
    "ww_2l2nu": "VV",
    "wz_2l2q": "VV",
    "wz_1l1nu2q": "VV",
    "wz_3lnu": "VV",
    "zz": "VV",
    "www": "VVV",
    "wwz": "VVV",
    "wzz": "VVV",
    "zzz": "VVV",
    "ggh_amcPS": "ggH",
    "vbf_powheg_dipole": "VBF",
}

grouping_alt = {
    "Data": [
        "data_A",
        "data_B",
        "data_C",
        "data_D",
        "data_E",
        "data_F",
        "data_G",
        "data_H",
    ],
    "DY": ["dy_m105_160_amc", "dy_m105_160_vbf_amc"],
    "EWK": ["ewk_lljj_mll105_160_ptj0"],
    "TT+ST": ["ttjets_dl", "ttjets_sl", "ttw", "ttz", "st_tw_top", "st_tw_antitop"],
    "VV": ["ww_2l2nu", "wz_2l2q", "wz_1l1nu2q", "wz_3lnu", "zz"],
    "VVV": ["www", "wwz", "wzz", "zzz"],
    "ggH": ["ggh_amcPS"],
    "VBF": ["vbf_powheg_dipole"],
}

decorrelation_scheme = {
    "LHERen": {"DY": ["DY"], "EWK": ["EWK"], "ggH": ["ggH"], "TT+ST": ["TT+ST"]},
    "LHEFac": {"DY": ["DY"], "EWK": ["EWK"], "ggH": ["ggH"], "TT+ST": ["TT+ST"]},
    "pdf_2rms": {"DY": ["DY"], "ggH": ["ggH"], "VBF": ["VBF"]},
    "pdf_mcreplica": {"DY": ["DY"], "ggH": ["ggH"], "VBF": ["VBF"]},
    # 'LHERen': {'DY':['DY_filter', 'DY_nofilter'], 'EWK':['EWK'],
    #            'ggH':['ggH'], 'TT+ST':['TT+ST']},
    # 'LHEFac': {'DY':['DY_filter', 'DY_nofilter'], 'EWK':['EWK'],
    #            'ggH':['ggH'], 'TT+ST':['TT+ST']},
    # 'pdf_2rms': {'DY':['DY_filter', 'DY_nofilter'],
    #             'ggH':['ggH'], 'VBF':['VBF']},
    # 'pdf_mcreplica': {'DY':['DY_filter', 'DY_nofilter'],
    #                  'ggH':['ggH'], 'VBF':['VBF']},
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
    df = df.repartition(npartitions=parameters["ncpus"])
    timer.add_checkpoint("Combined into a single Dask DataFrame")

    keep_columns = ["s", "year", "r"]
    keep_columns += [f"c {v}" for v in parameters["syst_variations"]]
    keep_columns += [c for c in df.columns if "wgt_" in c]
    keep_columns += parameters["hist_vars"]

    # Evaluate classifiers
    # TODO: outsource to GPUs
    evaluate_mva = True
    if evaluate_mva:
        for v in parameters["syst_variations"]:
            for model in parameters["dnn_models"]:
                score_name = f"score_{model} {v}"
                keep_columns += [score_name]
                df[score_name] = df.map_partitions(
                    dnn_evaluation, v, model, parameters, meta=(score_name, float)
                )
                timer.add_checkpoint(f"Evaluated {model} {v}")
            for model in parameters["bdt_models"]:
                score_name = f"score_{model} {v}"
                keep_columns += [score_name]
                df[score_name] = df.map_partitions(
                    bdt_evaluation, v, model, parameters, meta=(score_name, float)
                )
                timer.add_checkpoint(f"Evaluated {model} {v}")
    df = df[[c for c in keep_columns if c in df.columns]]
    df = df.compute()
    df.dropna(axis=1, inplace=True)
    df.reset_index(inplace=True)
    timer.add_checkpoint("Prepared for histogramming")

    argsets = []
    for var in parameters["hist_vars"]:
        for s in df.s.unique():
            argsets.append({"var": var, "s": s})
    # Make histograms
    hist_futures = client.map(partial(histogram, df=df, parameters=parameters), argsets)
    hists_ = client.gather(hist_futures)
    hists = {}
    for h in hists_:
        hists.update(h)
    timer.add_checkpoint("Histogramming")


def plotter(client, parameters, timer):
    # Load histograms
    argsets = []
    for year in parameters["years"]:
        for var in parameters["hist_vars"]:
            for s in parameters["samples"]:
                argsets.append({"var": var, "s": s, "year": year})
    hist_futures = client.map(partial(load_histograms, parameters=parameters), argsets)
    hists_ = client.gather(hist_futures)
    hists = {}
    # for h in hists_:
    # hists.update(h)
    for hist in hists_:
        for k1, h1 in hist.items():
            if k1 not in hists.keys():
                hists[k1] = {}
            for k2, h2 in h1.items():
                if k2 not in hists[k1].keys():
                    hists[k1][k2] = {}
                for k3, h3 in h2.items():
                    hists[k1][k2][k3] = h3
    timer.add_checkpoint("Loading histograms")

    # Plot histograms
    hists_to_plot = [
        hist for var, hist in hists.items() if var in parameters["plot_vars"]
    ]

    plot_futures = client.map(partial(plot, parameters=parameters), hists_to_plot)
    client.gather(plot_futures)

    timer.add_checkpoint("Plotting")


def load_data(path):
    if len(path) > 0:
        df = dd.read_parquet(path)
    else:
        df = dd.from_pandas(pd.DataFrame(), npartitions=1)

    return df


def prepare_features(df, parameters, variation="nominal", add_year=True):
    global training_features
    if add_year:
        features = training_features + ["year"]
    else:
        features = training_features
    features_var = []
    for trf in features:
        if f"{trf} {variation}" in df.columns:
            features_var.append(f"{trf} {variation}")
        elif trf in df.columns:
            features_var.append(trf)
        else:
            print(f"Variable {trf} not found in training dataframe!")
    return features_var


def dnn_evaluation(df, variation, model, parameters):
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        allow_soft_placement=True,
        device_count={"CPU": 1},
    )
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    sess = tf.compat.v1.Session(config=config)
    if parameters["do_massscan"]:
        mass_shift = parameters["mass"] - 125.0
    features = prepare_features(df, parameters, variation, add_year=True)
    score_name = f"score_{model} {variation}"
    try:
        df = df.compute()
    except Exception:
        pass
    df.loc[:, score_name] = 0
    with sess:
        nfolds = 4
        for i in range(nfolds):
            # FIXME
            label = f"allyears_jul7_{i}"

            # train_folds = [(i + f) % nfolds for f in [0, 1]]
            # val_folds = [(i + f) % nfolds for f in [2]]
            eval_folds = [(i + f) % nfolds for f in [3]]

            eval_filter = df.event.mod(nfolds).isin(eval_folds)

            scalers_path = f"{parameters['models_path']}/{model}/scalers_{label}.npy"
            scalers = np.load(scalers_path)
            model_path = f"{parameters['models_path']}/{model}/dnn_{label}.h5"
            dnn_model = load_model(model_path)
            df_i = df.loc[eval_filter, :]
            df_i.loc[df_i.r != "h-peak", "dimuon_mass"] = 125.0
            if parameters["do_massscan"]:
                df_i.loc[:, "dimuon_mass"] = df_i["dimuon_mass"] - mass_shift
            df_i = (df_i[features] - scalers[0]) / scalers[1]
            prediction = np.array(dnn_model.predict(df_i)).ravel()
            df.loc[eval_filter, score_name] = np.arctanh((prediction))
    return df[score_name]


def bdt_evaluation(df, variation, model, parameters):
    import pickle

    if parameters["do_massscan"]:
        mass_shift = parameters["mass"] - 125.0
    features = prepare_features(df, parameters, variation, add_year=False)
    score_name = f"score_{model} {variation}"
    try:
        df = df.compute()
    except Exception:
        pass
    df.loc[:, score_name] = 0
    nfolds = 4
    for i in range(nfolds):
        # FIXME
        label = f"2016_jul7_{i}"

        # train_folds = [(i + f) % nfolds for f in [0, 1]]
        # val_folds = [(i + f) % nfolds for f in [2]]
        eval_folds = [(i + f) % nfolds for f in [3]]

        eval_filter = df.event.mod(nfolds).isin(eval_folds)
        scalers_path = f"{parameters['models_path']}/{model}/scalers_{label}.npy"
        scalers = np.load(scalers_path)
        model_path = (
            f"{parameters['models_path']}/{model}/" f"BDT_model_earlystop50_{label}.pkl"
        )

        bdt_model = pickle.load(open(model_path, "rb"))
        df_i = df[eval_filter]
        df_i.loc[df_i.r != "h-peak", "dimuon_mass"] = 125.0
        if parameters["do_massscan"]:
            df_i.loc[:, "dimuon_mass"] = df_i["dimuon_mass"] - mass_shift
        df_i = (df_i[features] - scalers[0]) / scalers[1]
        if len(df_i) > 0:
            if "multiclass" in model:
                prediction = np.array(
                    bdt_model.predict_proba(df_i.values)[:, 5]
                ).ravel()
            else:
                prediction = np.array(
                    bdt_model.predict_proba(df_i.values)[:, 1]
                ).ravel()
            df.loc[eval_filter, score_name] = np.arctanh((prediction))
    return df[score_name]


def histogram(args, df=pd.DataFrame(), parameters={}):
    var = args["var"]
    s = args["s"]
    if var in variables_lookup.keys():
        var = variables_lookup[var]
    else:
        var = Variable(var, var, 50, 0, 5)

    # samples = df.s.unique()
    years = df.year.unique()
    regions = parameters["regions"]
    categories = parameters["categories"]
    syst_variations = parameters["syst_variations"]
    wgt_variations = [w for w in df.columns if ("wgt_" in w)]

    regions = [r for r in regions if r in df.r.unique()]
    categories = [c for c in categories if c in df["c nominal"].unique()]

    # sometimes different years have different binnings (MVA score)
    h = {}

    for year in years:
        h[year] = {}
        if "score" in var.name:
            bins = parameters["mva_bins"][var.name.replace("score_", "")][f"{year}"]
            h[year] = (
                Hist.new.StrCat(regions, name="region")
                .StrCat(categories, name="category")
                .StrCat(syst_variations, name="variation")
                .StrCat(wgt_variations, name="wgt_variation")
                .StrCat(["value", "sumw2"], name="val_err")
                .Var(bins, name=var.name)
                .Double()
            )
            # nbins = len(bins) - 1
        else:
            h[year] = (
                Hist.new.StrCat(regions, name="region")
                .StrCat(categories, name="category")
                .StrCat(syst_variations, name="variation")
                .StrCat(wgt_variations, name="wgt_variation")
                .StrCat(["value", "sumw2"], name="val_sumw2")
                .Reg(var.nbins, var.xmin, var.xmax, name=var.name, label=var.caption)
                .Double()
            )
            # nbins = var.nbins

        for r in regions:
            for v in syst_variations:
                varname = f"{var.name} {v}"
                if varname not in df.columns:
                    if var.name in df.columns:
                        varname = var.name
                    else:
                        continue
                for c in categories:
                    for w in wgt_variations:
                        slicer = (
                            (df.s == s)
                            & (df.r == r)
                            & (df.year == year)
                            & (df[f"c {v}"] == c)
                        )
                        data = df.loc[slicer, varname]
                        weight = df.loc[slicer, w]
                        h[year].fill(r, c, v, w, "value", data, weight=weight)
                        h[year].fill(r, c, v, w, "sumw2", data, weight=weight * weight)
                        # TODO: add treatment of PDF systematics
                        # (MC replicas)
        save_hist(h[year], var.name, s, year, parameters)
    return {var.name: {s: h}}


def save_hist(hist, var_name, s, year, parameters):
    # print(hist['h-peak', 'vbf', 'nominal', 'value', :].project(var_name).view())
    mkdir(parameters["hist_path"])
    hist_path = parameters["hist_path"] + parameters["label"]
    mkdir(hist_path)
    mkdir(f"{hist_path}/{year}")
    path = f"{hist_path}/{year}/{var_name}_{s}.pickle"
    with open(path, "wb") as handle:
        pickle.dump(hist, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_histograms(argset, parameters):
    var_name = argset["var"]
    s = argset["s"]
    year = argset["year"]
    hist_path = parameters["hist_path"] + parameters["label"]
    path = f"{hist_path}/{year}/{var_name}_{s}.pickle"
    try:
        with open(path, "rb") as handle:
            hist = pickle.load(handle)
    except Exception:
        return {}
    return {var_name: {year: {s: hist}}}


def plot(hist, parameters={}):
    if not hist.keys():
        return
    a_year = list(hist.keys())[0]
    a_sample = list(hist[a_year].keys())[0]
    var = hist[a_year][a_sample].axes[-1]
    plotsize = 8
    ratio_plot_size = 0.25

    r = "h-peak"
    c = "vbf"
    v = "nominal"
    w = "wgt_nominal"
    stack_groups = ["DY", "EWK", "TT+ST", "VV", "VVV"]
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
                if (g in self.groups) and (e in parameters["samples"])
            }
            self.entry_list = self.entry_dict.keys()
            self.labels = self.entry_dict.values()
            self.groups = list(set(self.entry_dict.values()))
            # print(self.entry_type, self.labels, self.groups)

        def get_plottables(self, hist, year, r, c, v, w, var_name):
            plottables = []
            sumw2 = []
            labels = []
            for group in self.groups:
                group_entries = [e for e, g in self.entry_dict.items() if (group == g)]
                if sum(
                    [
                        hist[year][en][r, c, v, w, "value", :].project(var_name)
                        for en in group_entries
                        if en in hist[year].keys()
                    ]
                ):
                    plottables.append(
                        sum(
                            [
                                hist[year][en][r, c, v, w, "value", :].project(var_name)
                                for en in group_entries
                                if en in hist[year].keys()
                            ]
                        )
                    )
                    sumw2.append(
                        sum(
                            [
                                hist[year][en][r, c, v, w, "sumw2", :].project(var_name)
                                for en in group_entries
                                if en in hist[year].keys()
                            ]
                        )
                    )
                    labels.append(group)
                    # print(group, sum(sum([hist[year][en][r, c, v, 'value', :]
                    #         .project(var_name) for en in group_entries if en in hist[year].keys()]).values()))
            return plottables, sumw2, labels

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
    for year in parameters["years"]:
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
            plottables, sumw2, labels = entry.get_plottables(
                hist, year, r, c, v, w, var.name
            )
            yerr = np.sqrt(sum(plottables).values()) if entry.yerr else None
            # print("#"*40)
            if len(plottables) == 0:
                continue
            # if entry.entry_type == 'step':
            #    print(plottables)
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
            num, _, _ = entries["data"].get_plottables(hist, year, r, c, v, w, var.name)
            num = sum(num).values()

        if len(entries["stack"].entry_list) > 0:
            # get MC yields and sumw2
            den, den_sumw2, _ = entries["stack"].get_plottables(
                hist, year, r, c, v, w, var.name
            )
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

        hep.cms.label(ax=plt1, data=True, label="Preliminary", year=year)

        path = parameters["plots_path"]
        out_name = f"{path}/{var.name}_{year}.png"
        fig.savefig(out_name)
        print(f"Saved: {out_name}")
    return
