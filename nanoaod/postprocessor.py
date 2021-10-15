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
from nanoaod.config.variables import variables_lookup, Variable
from python.utils import load_from_parquet
from python.utils import save_hist, load_histograms

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


def workflow(client, paths, parameters, timer=None):
    # Load dataframes
    df_future = client.map(load_from_parquet, paths)
    df_future = client.gather(df_future)
    if timer:
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
    if npart > 2 * parameters["ncpus"]:
        df = df.repartition(npartitions=parameters["ncpus"])
    if timer:
        timer.add_checkpoint("Combined into a single Dask DataFrame")

    # temporary
    if ("dataset" not in df.columns) and ("s" in df.columns):
        df["dataset"] = df["s"]
    if ("region" not in df.columns) and ("r" in df.columns):
        df["region"] = df["r"]
    for v in parameters["syst_variations"]:
        if (f"channel {v}" not in df.columns) and (f"c {v}" in df.columns):
            df[f"channel {v}"] = df[f"c {v}"]

    keep_columns = ["dataset", "year", "region"]
    keep_columns += [f"channel {v}" for v in parameters["syst_variations"]]
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
                if timer:
                    timer.add_checkpoint(f"Evaluated {model} {v}")
            for model in parameters["bdt_models"]:
                score_name = f"score_{model} {v}"
                keep_columns += [score_name]
                df[score_name] = df.map_partitions(
                    bdt_evaluation, v, model, parameters, meta=(score_name, float)
                )
                if timer:
                    timer.add_checkpoint(f"Evaluated {model} {v}")
    df = df[[c for c in keep_columns if c in df.columns]]
    df = df.compute()
    df.dropna(axis=1, inplace=True)
    df.reset_index(inplace=True)
    if timer:
        timer.add_checkpoint("Prepared for histogramming")

    argsets = []
    for year in df.year.unique():
        for var_name in parameters["hist_vars"]:
            for dataset in df.dataset.unique():
                argsets.append({"year": year, "var_name": var_name, "dataset": dataset})
    # Make histograms
    hist_futures = client.map(partial(histogram, df=df, parameters=parameters), argsets)
    hist_rows = client.gather(hist_futures)
    hist_df = pd.concat(hist_rows).reset_index(drop=True)
    if timer:
        timer.add_checkpoint("Histogramming")
    return hist_df


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
    if df.shape[0] == 0:
        return df[score_name]

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
            if df_i.shape[0] == 0:
                continue
            df_i.loc[df_i.region != "h-peak", "dimuon_mass"] = 125.0
            if parameters["do_massscan"]:
                df_i.loc[:, "dimuon_mass"] = df_i["dimuon_mass"] - mass_shift
            df_i = (df_i[features] - scalers[0]) / scalers[1]
            prediction = np.array(dnn_model.predict(df_i)).ravel()
            df.loc[eval_filter, score_name] = np.arctanh((prediction))
    return df[score_name]


def bdt_evaluation(df, variation, model, parameters):
    if parameters["do_massscan"]:
        mass_shift = parameters["mass"] - 125.0
    features = prepare_features(df, parameters, variation, add_year=False)
    score_name = f"score_{model} {variation}"
    try:
        df = df.compute()
    except Exception:
        pass
    df.loc[:, score_name] = 0
    if df.shape[0] == 0:
        return df[score_name]
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
        if df_i.shape[0] == 0:
            continue
        df_i.loc[df_i.region != "h-peak", "dimuon_mass"] = 125.0
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


def get_variation(wgt_variation, sys_variation):
    if "nominal" in wgt_variation:
        if "nominal" in sys_variation:
            return "nominal"
        else:
            return sys_variation
    else:
        if "nominal" in sys_variation:
            return wgt_variation
        else:
            return None


def histogram(args, df=pd.DataFrame(), parameters={}):
    year = args["year"]
    var_name = args["var_name"]
    dataset = args["dataset"]
    if var_name in variables_lookup.keys():
        var = variables_lookup[var_name]
    else:
        var = Variable(var_name, var_name, 50, 0, 5)

    regions = parameters["regions"]
    channels = parameters["channels"]
    wgt_variations = [w for w in df.columns if ("wgt_" in w)]
    syst_variations = parameters["syst_variations"]

    variations = []
    for w in wgt_variations:
        for v in syst_variations:
            variation = get_variation(w, v)
            if variation:
                variations.append(variation)

    regions = [r for r in regions if r in df.region.unique()]
    channels = [c for c in channels if c in df["channel nominal"].unique()]

    # sometimes different years have different binnings (MVA score)
    if "score" in var.name:
        bins = parameters["mva_bins"][var.name.replace("score_", "")][f"{year}"]
        hist = (
            Hist.new.StrCat(regions, name="region")
            .StrCat(channels, name="channel")
            .StrCat(variations, name="variation")
            .StrCat(["value", "sumw2"], name="val_err")
            .Var(bins, name=var.name)
            .Double()
        )
    else:
        hist = (
            Hist.new.StrCat(regions, name="region")
            .StrCat(channels, name="channel")
            .StrCat(variations, name="variation")
            .StrCat(["value", "sumw2"], name="val_sumw2")
            .Reg(var.nbins, var.xmin, var.xmax, name=var.name, label=var.caption)
            .Double()
        )

    for region in regions:
        for w in wgt_variations:
            for v in syst_variations:
                variation = get_variation(w, v)
                if not variation:
                    continue
                var_name = f"{var.name} {v}"
                if var_name not in df.columns:
                    if var.name in df.columns:
                        var_name = var.name
                    else:
                        continue
                for channel in channels:
                    slicer = (
                        (df.dataset == dataset)
                        & (df.region == region)
                        & (df.year == year)
                        & (df[f"channel {v}"] == channel)
                    )
                    data = df.loc[slicer, var_name]
                    weight = df.loc[slicer, w]
                    hist.fill(region, channel, variation, "value", data, weight=weight)
                    hist.fill(
                        region,
                        channel,
                        variation,
                        "sumw2",
                        data,
                        weight=weight * weight,
                    )
                    # TODO: add treatment of PDF systematics
                    # (MC replicas)
    if parameters["save_hists"]:
        save_hist(hist, var.name, dataset, year, parameters)
    hist_row = pd.DataFrame(
        [{"year": year, "var_name": var.name, "dataset": dataset, "hist": hist}]
    )
    return hist_row


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
                if (g in self.groups) and (e in parameters["datasets"])
            }
            self.entry_list = self.entry_dict.keys()
            self.labels = self.entry_dict.values()
            self.groups = list(set(self.entry_dict.values()))

        def get_plottables(self, hist, year, region, channel, variation, var_name):
            plottables_df = pd.DataFrame(columns=["label", "hist", "sumw2", "integral"])
            for group in self.groups:
                group_entries = [e for e, g in self.entry_dict.items() if (group == g)]
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

    fig.clf()
    fig.set_size_inches(plotsize * 1.2, plotsize * (1 + ratio_plot_size))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[(1 - ratio_plot_size), ratio_plot_size], hspace=0.07
    )

    # Top panel: Data/MC
    plt1 = fig.add_subplot(gs[0])

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

    if parameters["save_plots"]:
        path = parameters["plots_path"]
        out_name = f"{path}/{var.name}_{year}.png"
        fig.savefig(out_name)
        print(f"Saved: {out_name}")

    return total_yield
