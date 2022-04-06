import sys
import os
import pandas as pd
import multiprocessing as mp
import coffea
from coffea import util
import glob
import copy
import boost_histogram as bh
from boost_histogram import loc
import numpy as np
import uproot
from uproot_methods.classes.TH1 import from_numpy
import matplotlib.pyplot as plt
import mplhep as hep
import tqdm
from sklearn.metrics import roc_curve
from math import sqrt

stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
import keras  # noqa: E402

sys.stderr = stderr
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
pd.options.mode.chained_assignment = None

__all__ = ["keras"]

training_features_ = [
    "dimuon_mass",
    "dimuon_pt",
    "dimuon_eta",
    "dimuon_dEta",
    "dimuon_dPhi",
    "dimuon_dR",
    "jj_mass",
    "jj_eta",
    "jj_phi",
    "jj_pt",
    "jj_dEta",
    "mmjj_mass",
    "mmjj_eta",
    "mmjj_phi",
    "zeppenfeld",
    "jet1_pt",
    "jet1_eta",
    "jet1_qgl",
    "jet2_pt",
    "jet2_eta",
    "jet2_qgl",
    "dimuon_cosThetaCS",
    "dimuon_mass_res_rel",
    "mmj1_dEta",
    "mmj1_dPhi",
    "mmj2_dEta",
    "mmj2_dPhi",
    "htsoft5",
]

# Pisa variables
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

grouping = {
    "data_A": "Data",
    "data_B": "Data",
    "data_C": "Data",
    "data_D": "Data",
    "data_E": "Data",
    "data_F": "Data",
    "data_G": "Data",
    "data_H": "Data",
    "dy_0j": "DY",
    "dy_1j": "DY",
    "dy_2j": "DY",
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

bkg = ["DY", "EWK", "TT+ST", "VV"]
sig = ["VBF", "ggH"]

signal = {"vbf_powhegPS", "vbf_powheg_herwig", "vbf_powheg_dipole", "ggh_amcPS"}

rate_syst_lookup = {
    "2016": {
        # 'XsecAndNorm2016DYJ2': {
        #     'DYJ2_nofilter': 1.1291, 'DYJ2_filter': 1.12144},
        "XsecAndNorm2016DYJ2": {"DYJ2": 1.1291},
        "XsecAndNorm2016EWK": {"EWK": 1.06131},
        "XsecAndNormTT+ST": {"TT+ST": 1.182},
        "XsecAndNormVV": {"VV": 1.13203},
        "XsecAndNormggH": {"ggH_hmm": 1.38206},
    },
    "2017": {
        # 'XsecAndNorm2017DYJ2': {
        #     'DYJ2_nofilter': 1.13020, 'DYJ2_filter': 1.12409},
        "XsecAndNorm2017DYJ2": {"DYJ2": 1.13020},
        "XsecAndNorm2017EWK": {"EWK": 1.05415},
        "XsecAndNormTT+ST": {"TT+ST": 1.18406},
        "XsecAndNormVV": {"VV": 1.05653},
        "XsecAndNormggH": {"ggH_hmm": 1.37126},
    },
    "2018": {
        # 'XsecAndNorm2018DYJ2':{
        #     'DYJ2_nofilter': 1.12320, 'DYJ2_filter': 1.12077},
        "XsecAndNorm2018DYJ2": {"DYJ2": 1.12320},
        "XsecAndNorm2018EWK": {"EWK": 1.05779},
        "XsecAndNormTT+ST": {"TT+ST": 1.18582},
        "XsecAndNormVV": {"VV": 1.05615},
        "XsecAndNormggH": {"ggH_hmm": 1.38313},
    },
}

# https://twiki.cern.ch/twiki/bin/view/CMS/TWikiLUM
lumi_syst = {
    "2016": {
        "uncor": 2.2,
        "xyfac": 0.9,
        "len": 0.0,
        "bb": 0.4,
        "beta": 0.5,
        "calib": 0.0,
        "ghost": 0.4,
    },
    "2017": {
        "uncor": 2.0,
        "xyfac": 0.8,
        "len": 0.3,
        "bb": 0.4,
        "beta": 0.5,
        "calib": 0.3,
        "ghost": 0.1,
    },
    "2018": {
        "uncor": 1.5,
        "xyfac": 2.0,
        "len": 0.2,
        "bb": 0.0,
        "beta": 0.0,
        "calib": 0.2,
        "ghost": 0.0,
    },
}

decorrelation_scheme = {
    "LHERen": {"DY": ["DY"], "EWK": ["EWK"], "ggH": ["ggH"], "TT+ST": ["TT+ST"]},
    "LHEFac": {"DY": ["DY"], "EWK": ["EWK"], "ggH": ["ggH"], "TT+ST": ["TT+ST"]},
    "pdf_2rms": {"DY": ["DY"], "ggH": ["ggH"], "VBF": ["VBF"]},
    "pdf_mcreplica": {"DY": ["DY"], "ggH": ["ggH"], "VBF": ["VBF"]},
    # 'LHERen': {'DY': ['DY_filter', 'DY_nofilter'], 'EWK': ['EWK'],
    #            'ggH': ['ggH'], 'TT+ST': ['TT+ST']},
    # 'LHEFac': {'DY': ['DY_filter', 'DY_nofilter'], 'EWK': ['EWK'],
    #            'ggH': ['ggH'], 'TT+ST': ['TT+ST']},
    # 'pdf_2rms': {'DY': ['DY_filter', 'DY_nofilter'],
    #              'ggH': ['ggH'], 'VBF': ['VBF']},
    # 'pdf_mcreplica': {'DY': ['DY_filter', 'DY_nofilter'],
    #                   'ggH': ['ggH'], 'VBF': ['VBF']},
}

shape_only = ["LHE", "qgl"]


def worker(args):
    if "to_pandas" not in args["modules"]:
        print("Need to convert to Pandas DF first!")
        return
    df = to_pandas(args)
    if "evaluation" in args["modules"]:
        df = evaluation(df, args)
    hists = {}
    edges = {}
    if "get_hists" in args["modules"]:
        for var in args["vars_to_plot"]:
            hists[var.name], edges[var.name] = get_hists(
                df, var, args, mva_bins=args["mva_bins"]
            )
    return df, hists, edges


def postprocess(args, parallelize=True):
    dataframes = []
    hist_dfs = {}
    edges_dict = {}
    path = args["in_path"]
    if args["year"] == "":
        print("Loading samples for ALL years")
        years = ["2016", "2017", "2018"]
    else:
        years = [args["year"]]
    argsets = []
    all_training_samples = []
    classes_dict = {}
    if "training_samples" in args:
        for cl, sm in args["training_samples"].items():
            all_training_samples.extend(sm)
            for smp in sm:
                classes_dict[smp] = cl
    args.update({"classes_dict": classes_dict})
    for year in years:
        for s in args["samples"]:
            variations = args["syst_variations"]
            for v in variations:
                proc_outs = glob.glob(f"{path}/{year}_{args['label']}/{v}/{s}.coffea")
                if len(proc_outs) == 0:
                    proc_outs = glob.glob(
                        f"{path}/{year}_{args['label']}" f"/nominal/{s}.coffea"
                    )
                for proc_path in proc_outs:
                    for c in args["channels"]:
                        for r in args["regions"]:
                            argset = args.copy()
                            argset.update(
                                **{
                                    "proc_path": proc_path,
                                    "s": s,
                                    "c": c,
                                    "r": r,
                                    "v": v,
                                    "y": year,
                                }
                            )
                            argsets.append(argset)
    if len(argsets) == 0:
        print("Nothing to load! Check the arguments.")
        sys.exit(0)

    if parallelize:
        cpus = mp.cpu_count() - 2
        print(f"Parallelizing over {cpus} CPUs")

        pbar = tqdm.tqdm(total=len(argsets))

        def update(*a):
            pbar.update()

        pool = mp.Pool(cpus)
        a = [
            pool.apply_async(worker, args=(argset,), callback=update)
            for argset in argsets
        ]
        for process in a:
            process.wait()
            df, hists, edges = process.get()
            dataframes.append(df)
            for var, hist in hists.items():
                if var in edges_dict.keys():
                    if edges_dict[var] == []:
                        edges_dict[var] = edges[var]
                else:
                    edges_dict[var] = edges[var]
                if var in hist_dfs.keys():
                    hist_dfs[var].append(hist)
                else:
                    hist_dfs[var] = [hist]
        pool.close()
    else:
        for argset in argsets:
            df, hists, edges = worker(argset)
            dataframes.append(df)
            for var, hist in hists.items():
                if var in edges_dict.keys():
                    if edges_dict[var] == []:
                        edges_dict[var] = edges[var]
                else:
                    edges_dict[var] = edges[var]
                if var in hist_dfs.keys():
                    hist_dfs[var].append(hist)
                else:
                    hist_dfs[var] = [hist]
    return dataframes, hist_dfs, edges_dict


# for unbinned processor output (column_accumulators)
def to_pandas(args):
    proc_out = util.load(args["proc_path"])
    c = args["c"]
    r = args["r"]
    s = args["s"]
    v = args["v"]
    year = args["y"]
    suff = f"_{c}_{r}"

    # groups = ['DY_nofilter', 'DY_filter', 'EWK',
    #           'TT+ST', 'VV', 'ggH', 'VBF']
    # groups = ['DY', 'EWK', 'TT+ST', 'VV', 'ggH', 'VBF']
    columns = [c.replace(suff, "") for c in list(proc_out.keys()) if suff in c]
    df = pd.DataFrame()

    if f"dimuon_mass_{c}_{r}" in proc_out.keys():
        len_ = len(proc_out[f"dimuon_mass_{c}_{r}"].value)
    else:
        len_ = 0

    for var in columns:
        # if DNN training: get only relevant variables
        if args["training"] and (
            var not in training_features + ["event", "wgt_nominal"]
        ):
            continue
        # possibility to ignore weight variations
        if (not args["wgt_variations"]) and ("wgt_" in var) and ("nominal" not in var):
            continue
        # for JES/JER systematic variations do not consider weight variations
        if (v != "nominal") and ("wgt_" in var) and ("nominal" not in var):
            continue
        # Theory uncertainties only for VBF samples
        if s in grouping.keys():
            if (grouping[s] != "VBF") and ("THU" in var):
                continue
        else:
            if "THU" in var:
                continue

        done = False
        for syst, decorr in decorrelation_scheme.items():
            if s not in grouping.keys():
                continue
            if "data" in s:
                continue
            if ("2016" in year) and ("pdf_2rms" in var):
                continue
            if ("2016" not in year) and ("pdf_mcreplica" in var):
                continue
            if syst in var:
                if "off" in var:
                    continue
                suff = ""
                if "_up" in var:
                    suff = "_up"
                elif "_down" in var:
                    suff = "_down"
                else:
                    continue
                vname = var.replace(suff, "")
                for dec_group, proc_groups in decorr.items():
                    try:
                        if grouping[s] in proc_groups:
                            df[f"{vname}_{dec_group}{suff}"] = proc_out[
                                f"{var}_{c}_{r}"
                            ].value
                        else:
                            df[f"{vname}_{dec_group}{suff}"] = proc_out[
                                f"wgt_nominal_{c}_{r}"
                            ].value
                    except Exception:
                        df[f"{vname}_{dec_group}{suff}"] = proc_out[
                            f"wgt_nominal_{c}_{r}"
                        ].value
                    done = True

        if not done:
            try:
                if len(proc_out[f"{var}_{c}_{r}"].value) > 0:
                    df[var] = proc_out[f"{var}_{c}_{r}"].value
                else:
                    if "wgt_" in var:
                        df[var] = proc_out[f"wgt_nominal_{c}_{r}"].value
                    else:
                        df[var] = np.zeros(len_, dtype=float)
            except Exception:
                if "wgt_" in var:
                    df[var] = proc_out[f"wgt_nominal_{c}_{r}"].value
                else:
                    df[var] = np.zeros(len_, dtype=float)

    df["c"] = c
    df["r"] = r
    df["s"] = s
    df["v"] = v
    df["year"] = int(year)
    if args["training"]:
        if s in args["classes_dict"].keys():
            df["cls"] = args["classes_dict"][s]
        else:
            df["cls"] = ""
    else:
        if s in signal:
            df["cls"] = "signal"
        else:
            df["cls"] = "background"
    if ("extra_events" in args.keys()) and ("plot_extra" in args.keys()):
        if ("data" in s) and (args["plot_extra"]):
            df = df[df["event"].isin(args["extra_events"])]
    return df


def prepare_features(df, args, add_year=False):
    global training_features
    features = copy.deepcopy(training_features)
    df["dimuon_pt_log"] = np.log(df["dimuon_pt"])
    df["jj_mass_log"] = np.log(df["jj_mass"])
    if add_year and ("year" not in features):
        features += ["year"]
    if not add_year and ("year" in features):
        features = [t for t in features if t != "year"]
    for trf in features:
        if trf not in df.columns:
            print(f"Variable {trf} not found in training dataframe!")
    return df, features


def dnn_training(df, args, model):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import BatchNormalization

    def scale_data(inputs, model, label):
        x_mean = np.mean(x_train[inputs].values, axis=0)
        x_std = np.std(x_train[inputs].values, axis=0)
        training_data = (x_train[inputs] - x_mean) / x_std
        validation_data = (x_val[inputs] - x_mean) / x_std
        np.save(f"output/trained_models/{model}/scalers_{label}", [x_mean, x_std])
        return training_data, validation_data

    nfolds = 4
    classes = df.cls.unique()
    cls_idx_map = {cls: idx for idx, cls in enumerate(classes)}
    add_year = args["year"] == ""
    df, features = prepare_features(df, args, add_year)
    df["cls_idx"] = df["cls"].map(cls_idx_map)

    try:
        os.mkdir(f"output/trained_models/{model}/")
    except Exception:
        pass

    print("Training features: ", features)
    for i in range(nfolds):
        if args["year"] == "":
            label = f"allyears_{args['label']}_{i}"
        else:
            label = f"allyears_{args['year']}_{args['label']}_{i}"

        train_folds = [(i + f) % nfolds for f in [0, 1]]
        val_folds = [(i + f) % nfolds for f in [2]]
        eval_folds = [(i + f) % nfolds for f in [3]]

        print(f"Train classifier #{i + 1} out of {nfolds}")
        print(f"Training folds: {train_folds}")
        print(f"Validation folds: {val_folds}")
        print(f"Evaluation folds: {eval_folds}")
        print("Samples used: ", df.s.unique())

        train_filter = df.event.mod(nfolds).isin(train_folds)
        val_filter = df.event.mod(nfolds).isin(val_folds)
        # eval_filter = df.event.mod(nfolds).isin(eval_folds)

        other_columns = ["event", "wgt_nominal"]

        df_train = df[train_filter]
        df_val = df[val_filter]

        x_train = df_train[features]
        y_train = df_train["cls_idx"]
        x_val = df_val[features]
        y_val = df_val["cls_idx"]

        df_train["cls_avg_wgt"] = 1.0
        df_val["cls_avg_wgt"] = 1.0
        for icls, cls in enumerate(classes):
            train_evts = len(y_train[y_train == icls])
            df_train.loc[y_train == icls, "cls_avg_wgt"] = df_train.loc[
                y_train == icls, "wgt_nominal"
            ].values.mean()
            df_val.loc[y_val == icls, "cls_avg_wgt"] = df_val.loc[
                y_val == icls, "wgt_nominal"
            ].values.mean()
            print(f"{train_evts} training events in class {cls}")

        for smp in df_train.s.unique():
            df_train.loc[df_train.s == smp, "smp_avg_wgt"] = df_train.loc[
                df_train.s == smp, "wgt_nominal"
            ].values.mean()
            df_val.loc[df_val.s == smp, "smp_avg_wgt"] = df_val.loc[
                df_val.s == smp, "wgt_nominal"
            ].values.mean()
            print(f"{train_evts} training events in class {cls}")

        # df_train['training_wgt'] =\
        #     df_train['wgt_nominal']/df_train['cls_avg_wgt']
        # df_val['training_wgt'] =\
        #     df_val['wgt_nominal']/df_val['cls_avg_wgt']
        df_train["training_wgt"] = df_train["wgt_nominal"] / df_train["smp_avg_wgt"]
        df_val["training_wgt"] = df_val["wgt_nominal"] / df_val["smp_avg_wgt"]

        # scale data
        x_train, x_val = scale_data(features, model, label)
        x_train[other_columns] = df_train[other_columns]
        x_val[other_columns] = df_val[other_columns]

        # load model
        input_dim = len(features)
        inputs = Input(shape=(input_dim,), name=label + "_input")
        x = Dense(128, name=label + "_layer_1", activation="tanh")(inputs)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(64, name=label + "_layer_2", activation="tanh")(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(32, name=label + "_layer_3", activation="tanh")(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        # x = Dense(8, name=label + '_layer_4', activation='tanh')(x)
        # x = Dropout(0.2)(x)
        # x = BatchNormalization()(x)
        # x = Dense(8, name=label + '_layer_5', activation='tanh')(x)
        # x = Dropout(0.2)(x)
        # x = BatchNormalization()(x)
        outputs = Dense(1, name=label + "_output", activation="sigmoid")(x)

        dnn = Model(inputs=inputs, outputs=outputs)
        dnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        dnn.summary()

        # history = dnn.fit(
        #     x_train[features], y_train, epochs=100, batch_size=1024,
        #     sample_weight=df_train['training_wgt'].values, verbose=1,
        #     validation_data=(x_val[features], y_val,
        #     df_val['training_wgt'].values), shuffle=True)

        history = dnn.fit(
            x_train[features],
            y_train,
            epochs=100,
            batch_size=1024,
            verbose=1,
            validation_data=(x_val[features], y_val),
            shuffle=True,
        )

        util.save(
            history.history, f"output/trained_models/{model}/history_{label}.coffea"
        )
        dnn.save(f"output/trained_models/{model}/dnn_{label}.h5")


def evaluation(df, args):
    if df.shape[0] == 0:
        return df
    for model in args["dnn_models"]:
        df = dnn_evaluation(df, model, args)
    for model in args["bdt_models"]:
        df = bdt_evaluation(df, model, args)
    return df


def dnn_evaluation(df, model, args):
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        allow_soft_placement=True,
        device_count={"CPU": 1},
    )
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    sess = tf.compat.v1.Session(config=config)
    if args["do_massscan"]:
        mass_shift = args["mass"] - 125.0
    allyears = "allyears" in model
    df, features = prepare_features(df, args, allyears)
    score_name = f"score_{model}"
    df[score_name] = 0
    with sess:
        nfolds = 4
        for i in range(nfolds):
            if allyears:
                label = f"allyears_{args['label']}_{i}"
            else:
                label = f"{args['year']}_{args['label']}_{i}"

            # train_folds = [(i + f) % nfolds for f in [0, 1]]
            # val_folds = [(i + f) % nfolds for f in [2]]
            eval_folds = [(i + f) % nfolds for f in [3]]

            eval_filter = df.event.mod(nfolds).isin(eval_folds)

            scalers_path = f"output/trained_models/{model}/scalers_{label}.npy"
            scalers = np.load(scalers_path)
            model_path = f"output/trained_models/{model}/dnn_{label}.h5"
            dnn_model = load_model(model_path)
            df_i = df[eval_filter]
            if args["r"] != "h-peak":
                df_i["dimuon_mass"] = 125.0
            if args["do_massscan"]:
                df_i["dimuon_mass"] = df_i["dimuon_mass"] - mass_shift

            df_i = (df_i[features] - scalers[0]) / scalers[1]
            prediction = np.array(dnn_model.predict(df_i)).ravel()
            df.loc[eval_filter, score_name] = np.arctanh((prediction))
    return df


def bdt_evaluation(df, model, args):
    import pickle

    if args["do_massscan"]:
        mass_shift = args["mass"] - 125.0
    allyears = "allyears" in model
    df, features = prepare_features(df, args, allyears)
    score_name = f"score_{model}"
    df[score_name] = 0

    nfolds = 4
    for i in range(nfolds):
        if allyears:
            label = f"allyears_{args['label']}_{i}"
        else:
            label = f"{args['year']}_{args['label']}_{i}"

        # train_folds = [(i + f) % nfolds for f in [0, 1]]
        # val_folds = [(i + f) % nfolds for f in [2]]
        eval_folds = [(i + f) % nfolds for f in [3]]

        eval_filter = df.event.mod(nfolds).isin(eval_folds)

        scalers_path = f"output/trained_models/{model}/scalers_{label}.npy"
        scalers = np.load(scalers_path)
        model_path = f'output/trained_models/{model}/"\
            f"BDT_model_earlystop50_{label}.pkl'
        bdt_model = pickle.load(open(model_path, "rb"))
        df_i = df[eval_filter]
        if args["r"] != "h-peak":
            df_i["dimuon_mass"] = 125.0
        if args["do_massscan"]:
            df_i["dimuon_mass"] = df_i["dimuon_mass"] - mass_shift
        df_i = (df_i[features] - scalers[0]) / scalers[1]
        # prediction = np.array(
        #     bdt_model.predict_proba(df_i)[:, 1]).ravel()
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
    return df


def rebin(df, model, args):
    print("Rebinning DNN/BDT...")
    # Synchronize per-bin VBF yields with Pisa datacards
    df = df[(df.s == "vbf_powheg_dipole") & (df.r == "h-peak")]
    if args["year"] == "":
        years = ["2016", "2017", "2018"]
    else:
        years = [args["year"]]
    var = f"score_{model}"
    df = df.sort_values(by=[var], ascending=False)
    bnd = {}
    target_yields = {
        "2016": [
            0.39578,
            0.502294,
            0.511532,
            0.521428,
            0.529324,
            0.542333,
            0.550233,
            0.562859,
            0.572253,
            0.582248,
            0.588619,
            0.596933,
            0.606919,
        ],
        "2017": [
            0.460468,
            0.559333,
            0.578999,
            0.578019,
            0.580368,
            0.585521,
            0.576521,
            0.597367,
            0.593959,
            0.59949,
            0.595802,
            0.596376,
            0.57163,
        ],
        "2018": [
            0.351225,
            1.31698,
            1.25503,
            1.18703,
            1.12262,
            1.06208,
            0.995618,
            0.935661,
            0.86732,
            0.80752,
            0.73571,
            0.670533,
            0.608029,
        ],
    }
    target_yields["combined"] = []
    for i in range(len(target_yields["2016"])):
        target_yields["combined"].append(
            target_yields["2016"][i]
            + target_yields["2017"][i]
            + target_yields["2018"][i]
        )
    for year in years + ["combined"]:
        bnd[year] = {}
        for c in df.c.unique():
            bnd[year][c] = {}
            for v in df.v.unique():
                bin_sum = 0
                boundaries = []
                idx_left = 0
                idx_right = len(target_yields[year]) - 1
                if year == "combined":
                    filter = (df.c == c) & (df.v == v)
                else:
                    filter = (df.c == c) & (df.v == v) & (df.year == int(year))
                for idx, row in df[filter].iterrows():
                    bin_sum += row["wgt_nominal"]
                    if bin_sum >= target_yields[year][idx_right]:
                        boundaries.append(round(row[var], 3))
                        bin_sum = 0
                        idx_left += 1
                        idx_right -= 1
                bnd[year][c][v] = sorted([0, 5.0] + boundaries)
    # print(model, bnd)
    return bnd


def get_asimov_significance(df, model, args):
    if args["year"] == "":
        years = ["2016", "2017", "2018"]
    else:
        years = [args["year"]]
    binning_all = rebin(df, model, args)
    binning = {}
    significance2 = 0
    var = f"score_{model}"
    print(df[var].max())
    for year in years:
        binning = binning_all[year]["vbf"]["nominal"]
        S = df[(df.cls == "signal") & (df.r == "h-peak") & (df.year == int(year))]
        B = df[(df.cls == "background") & (df.r == "h-peak") & (df.year == int(year))]
        for ibin in range(len(binning) - 1):
            bin_lo = binning[ibin]
            bin_hi = binning[ibin + 1]
            sig_yield = S[(S[var] >= bin_lo) & (S[var] < bin_hi)]["wgt_nominal"].sum()
            bkg_yield = B[(B[var] >= bin_lo) & (B[var] < bin_hi)]["wgt_nominal"].sum()
            # print(f'{year} Bin {ibin}: s={sig_yield}, b={bkg_yield},"\
            #     f" sigma={sig_yield/sqrt(sig_yield+bkg_yield)}')
            significance2 += (sig_yield * sig_yield) / (sig_yield + bkg_yield)
    print(f"Model {model}: significance = {sqrt(significance2)}")
    return sqrt(significance2)


def plot_rocs(df, args):
    classes = df.cls.unique()
    cls_idx_map = {cls: idx for idx, cls in enumerate(classes)}
    df["cls_idx"] = df["cls"].map(cls_idx_map)
    fig = plt.figure()
    fig.set_size_inches(10, 8)
    plt.rcParams.update({"font.size": 15})
    for model in args["dnn_models"] + args["bdt_models"]:
        print(model)
        var = f"score_{model}"
        if var not in df.columns:
            continue
        significance = round(get_asimov_significance(df, model, args), 3)
        roc = roc_curve(df["cls_idx"], df[var], sample_weight=df["wgt_nominal"])
        plt.plot(roc[1], roc[0], label=f"{model} sign={significance}")
        # [0]: fpr, [1]: tpr, [2]: threshold
    plt.xlabel("Signal efficiency")
    plt.ylabel("Bkg efficiency")
    plt.legend(loc="best", fontsize=18)
    plt.ylim(0.0001, 0.1)
    plt.yscale("log")
    plt.xlim(0, 0.3)
    try:
        os.mkdir("plots_new/roc_curves")
    except Exception:
        pass
    save_path = f'plots_new/roc_curves/roc_{args["year"]}_{args["label"]}.png'
    fig.savefig(save_path)
    print(f"ROC curves saved to {save_path}")


def overlap_study(df, args, model):
    purdue_bins = rebin(df, model, args)[args["year"]]["vbf"]["nominal"]
    varname = f"score_{model}"
    if args["year"] == "2016":
        df = df[(df.s == "vbf_powheg_herwig") & (df.r == "h-peak")]
    else:
        df = df[(df.s == "vbf_powhegPS") & (df.r == "h-peak")]
    pisa_files = {
        "2016": "/depot/cms/hmm/pisa/vbfHmm_2016POWHERWIGSnapshot.root",
        "2017": "/depot/cms/hmm/pisa/vbfHmm_2017POWPYSnapshot.root",
        "2018": "/depot/cms/hmm/pisa/vbfHmm_2018POWPYSnapshot.root",
    }

    pisa_bins = {
        "2016": [
            0,
            0.846666666667,
            1.30333333333,
            1.575,
            1.815,
            2.03833333333,
            2.23666666667,
            2.41,
            2.575,
            2.74,
            2.91833333333,
            3.12333333333,
            3.4,
            5.0,
        ],
        "2017": [
            0,
            0.846666666667,
            1.30333333333,
            1.575,
            1.815,
            2.03833333333,
            2.23666666667,
            2.41,
            2.575,
            2.74,
            2.91833333333,
            3.12333333333,
            3.4,
            5.0,
        ],
        "2018": [
            0,
            0.623333333333,
            1.34166666667,
            1.70166666667,
            1.98333333333,
            2.22166666667,
            2.42,
            2.58666666667,
            2.74166666667,
            2.9,
            3.06166666667,
            3.24166666667,
            3.48,
            5.0,
        ],
    }

    with uproot.open(pisa_files[args["year"]]) as f:
        tree = f["Events"]
        pisa_dict = {
            "event": tree["event"].array(),
            "dnn_score": tree["DNN18Atan"].array(),
        }
        pisa_df = pd.DataFrame(pisa_dict)

    if len(purdue_bins) != len(pisa_bins[args["year"]]):
        print("Inconsistent number of bins!")
        return
    overlap = np.intersect1d(
        df.event.values.astype(int), pisa_df.event.values.astype(int)
    )
    # print(f'Total Purdue events: {df.shape[0]}')
    # print(f'Total Pisa events: {pisa_df.shape[0]}')
    # percent = round(100 * len(overlap) / pisa_df.shape[0], 2)
    # print(f'Common events: {len(overlap)} ({percent}%)')
    df = df.set_index("event")
    pisa_df = pisa_df.set_index("event")
    df_combined = pd.DataFrame(index=overlap)
    df_combined.loc[overlap, "purdue_score"] = df.loc[overlap, varname]
    df_combined.loc[overlap, "pisa_score"] = pisa_df.loc[overlap, "dnn_score"]

    narrow = 200
    options = {
        "analysis": {"purdue_bins": purdue_bins, "pisa_bins": pisa_bins[args["year"]]},
        "narrow": {
            "purdue_bins": [x * max(purdue_bins) / narrow for x in range(narrow + 1)],
            "pisa_bins": [
                x * max(pisa_bins[args["year"]]) / narrow for x in range(narrow + 1)
            ],
        },
    }

    df_combined["weight"] = 1.0
    for opt_name, binning in options.items():
        for i in range(len(purdue_bins) - 1):
            ibin = (df_combined.pisa_score >= pisa_bins[args["year"]][i]) & (
                df_combined.pisa_score < pisa_bins[args["year"]][i + 1]
            )
            if opt_name == "analysis":
                df_combined.loc[ibin, "weight"] = 1 / df_combined[ibin].shape[0]
            else:
                df_combined.loc[ibin, "weight"] = 1.0
        H, xedges, yedges = np.histogram2d(
            df_combined.purdue_score,
            df_combined.pisa_score,
            bins=(binning["purdue_bins"], binning["pisa_bins"]),
            weights=df_combined.weight.values,
        )

        fig = plt.figure()
        fig.set_size_inches(10, 10)
        plt.rcParams.update({"font.size": 15})
        ax = hep.hist2dplot(H, xedges, yedges)

        if opt_name == "analysis":
            for i in range(len(yedges) - 1):
                for j in range(len(xedges) - 1):
                    ax.text(
                        0.5 * (xedges[j] + xedges[j + 1]),
                        0.5 * (yedges[i] + yedges[i + 1]),
                        round(H.T[i, j], 2),
                        color="w",
                        ha="center",
                        va="center",
                        fontweight="bold",
                        fontsize=8,
                    )
        ax.set_xlabel("Purdue")
        ax.set_ylabel("Pisa")
        savename = (
            f'plots_new/plots_{args["year"]}_{args["label"]}/'
            f"overlap_{model}_{opt_name}.png"
        )
        fig.savefig(savename)


def overlap_study_unbinned(purdue_df, args, model):
    # scores = [c for c in purdue_df.columns if 'score' in c]
    var = f"score_{model}"
    mass_point = str(args["mass"]).replace(".", "")
    path = "/depot/cms/hmm/pisa/data_massScan/"
    pisa_files = {
        "2016": path + "data2016Snapshot.root",
        "2017": path + "data2017Snapshot.root",
        "2018": path + "data2018Snapshot.root",
    }
    pisa_bins = [
        0,
        0.846666666667,
        1.30333333333,
        1.575,
        1.815,
        2.03833333333,
        2.23666666667,
        2.41,
        2.575,
        2.74,
        2.91833333333,
        3.12333333333,
        3.4,
        5.0,
    ]
    # purdue_bins = rebin(
    #    purdue_df, model, args)['combined']['vbf']['nominal']
    # purdue_bins = [0, 0.147, 0.455, 0.703, 0.905, 1.08, 1.238,
    #        1.38, 1.515, 1.642, 1.759, 1.871, 1.984, 5.0]
    for c in purdue_df.columns:
        if "wgt" in c:
            continue
        if c not in training_features:
            continue
        print(c, purdue_df[purdue_df.event == 190674864.0][c].values[0])

    purdue_df = purdue_df[
        purdue_df.s.isin([s for s in purdue_df.s.unique() if "data" in s])
    ]
    df = pd.DataFrame()

    if args["year"] == "":
        years = ["2016", "2017", "2018"]
        purdue_bins = [
            0,
            0.147,
            0.455,
            0.703,
            0.905,
            1.08,
            1.238,
            1.38,
            1.515,
            1.642,
            1.759,
            1.871,
            1.984,
            5.0,
        ]
    else:
        years = [args["year"]]
        purdue_bins = args["mva_bins"][model][args["year"]]

    for year in years:
        with uproot.open(pisa_files[year]) as f:
            tree = f["Events"]
            pisa_dict = {
                "event": tree["event"].array(),
                "dnn_score_pisa": tree[f"DNN18AtanM{mass_point}"].array(),
            }
            new_df = pd.DataFrame(pisa_dict)
            if len(df) > 0:
                df = pd.concat([df, new_df])
            else:
                df = new_df
    df.set_index("event", inplace=True, drop=False)
    # purdue_df.event = purdue_df.event.astype(int)
    purdue_df.set_index("event", inplace=True, drop=False)
    purdue_df_overlap = purdue_df[
        (purdue_df.r == "h-peak") & purdue_df.event.isin(df.event)
    ]
    purdue_df_extra = purdue_df[
        (purdue_df.r == "h-peak") & ~purdue_df.event.isin(df.event)
    ]
    purdue_df_extra["dnn_score_purdue"] = purdue_df_extra[var]
    df_extra = purdue_df_extra[["event", "dnn_score_purdue"]]
    df_extra.set_index("event", inplace=True, drop=False)
    df_extra["dnn_score_pisa"] = 0

    df.loc[purdue_df_overlap["event"], "dnn_score_purdue"] = purdue_df_overlap[var]
    df = pd.concat([df, df_extra])
    df.fillna(0, inplace=True)

    # print(df[abs(df['dnn_score_purdue']-df['dnn_score_pisa'])>2])

    import matplotlib.pyplot as plt

    plt.style.use(hep.style.ROOT)
    f, ax = plt.subplots()
    x = df["dnn_score_purdue"]
    y = df["dnn_score_pisa"]

    from matplotlib.patches import Rectangle

    for i in range(len(pisa_bins) - 1):
        bin_length_x = purdue_bins[i + 1] - purdue_bins[i]
        bin_length_y = pisa_bins[i + 1] - pisa_bins[i]
        ax.add_patch(
            Rectangle(
                (purdue_bins[i], pisa_bins[i]),
                bin_length_x,
                bin_length_y,
                alpha=0.3,
                zorder=0,
            )
        )

    for ibin in pisa_bins:
        ax.plot([0, 2.8], [ibin, ibin], c="red", zorder=1)
    for ibin in purdue_bins:
        ax.plot([ibin, ibin], [0, 5], c="red", zorder=1)

    ax.scatter(x, y, s=5, c="black", zorder=2)

    ax.set_xlabel("Purdue DNN")
    ax.set_ylabel("Pisa DNN")
    if args["year"] == "":
        save_dir = f'plots_new/plots_combined_{args["label"]}/'
        save_name = f"overlap_unbinned_{model}_{mass_point}.png"
        try:
            os.mkdir(save_dir)
        except Exception:
            pass
    else:
        # save_dir = f'plots_new/plots_{args["year"]}_{args["label"]}/'
        save_dir = f'plots_new/plots_combined_{args["label"]}/'
        save_name = f'overlap_unbinned_{args["year"]}_{model}_{mass_point}.png'
    save_path = f"{save_dir}/{save_name}"
    print(f"Saving plot to {save_path}")
    f.savefig(save_path)


def get_hists(df, var, args, mva_bins=[]):
    if "score" in var.name:
        bins = mva_bins[var.name.replace("score_", "")][args["year"]]
    dataset_axis = bh.axis.StrCategory(df.s.unique())
    region_axis = bh.axis.StrCategory(df.r.unique())
    channel_axis = bh.axis.StrCategory(df.c.unique())
    syst_axis = bh.axis.StrCategory(df.v.unique())
    val_err_axis = bh.axis.StrCategory(["value", "sumw2"])
    if ("score" in var.name) and len(bins) > 0:
        var_axis = bh.axis.Variable(bins)
        nbins = len(bins) - 1
    else:
        var_axis = bh.axis.Regular(var.nbins, var.xmin, var.xmax)
        nbins = var.nbins
    df_out = pd.DataFrame()
    edges = []
    regions = df.r.unique()
    channels = df.c.unique()
    for s in df.s.unique():
        if "data" in s:
            if "nominal" in args["syst_variations"]:
                syst_variations = ["nominal"]
                wgts = ["wgt_nominal"]
            else:
                syst_variations = []
                wgts = []
        else:
            syst_variations = args["syst_variations"]
            wgts = [c for c in df.columns if ("wgt_" in c)]
        mcreplicas = [c for c in df.columns if ("mcreplica" in c)]
        mcreplicas = []
        if len(mcreplicas) > 0:
            wgts = [wgt for wgt in wgts if ("pdf_2rms" not in wgt)]
        if (
            len(mcreplicas) > 0
            and ("wgt_nominal" in df.columns)
            and (s in grouping.keys())
        ):
            decor = decorrelation_scheme["pdf_mcreplica"]
            for decor_group, proc_groups in decor.items():
                for imcr, mcr in enumerate(mcreplicas):
                    wgts += [f"pdf_mcreplica{imcr}_{decor_group}"]
                    if grouping[s] in proc_groups:
                        df.loc[:, f"pdf_mcreplica{imcr}_{decor_group}"] = np.multiply(
                            df.wgt_nominal, df[mcr]
                        )
                    else:
                        df.loc[:, f"pdf_mcreplica{imcr}_{decor_group}"] = df.wgt_nominal
        for w in wgts:
            hist = bh.Histogram(
                dataset_axis,
                region_axis,
                channel_axis,
                syst_axis,
                val_err_axis,
                var_axis,
            )
            hist.fill(
                df.s.to_numpy(),
                df.r.to_numpy(),
                df.c.to_numpy(),
                df.v.to_numpy(),
                "value",
                df[var.name].to_numpy(),
                weight=df[w].to_numpy(),
            )
            hist.fill(
                df.s.to_numpy(),
                df.r.to_numpy(),
                df.c.to_numpy(),
                df.v.to_numpy(),
                "sumw2",
                df[var.name].to_numpy(),
                weight=(df[w] * df[w]).to_numpy(),
            )
            for v in df.v.unique():
                if v not in syst_variations:
                    continue
                if (v != "nominal") & (w != "wgt_nominal"):
                    continue
                for r in regions:
                    for c in channels:
                        values = hist[
                            loc(s), loc(r), loc(c), loc(v), loc("value"), :
                        ].to_numpy()[0]
                        sumw2 = hist[
                            loc(s), loc(r), loc(c), loc(v), loc("sumw2"), :
                        ].to_numpy()[0]
                        values[values < 0] = 0
                        sumw2[values < 0] = 0
                        integral = values.sum()
                        edges = hist[
                            loc(s), loc(r), loc(c), loc(v), loc("value"), :
                        ].to_numpy()[1]
                        contents = {}
                        contents.update({f"bin{i}": [values[i]] for i in range(nbins)})
                        contents.update({"sumw2_0": 0.0})
                        # add a dummy bin b/c sumw2 indices
                        # are shifted w.r.t bin indices
                        contents.update(
                            {f"sumw2_{i + 1}": [sumw2[i]] for i in range(nbins)}
                        )
                        contents.update(
                            {
                                "s": [s],
                                "r": [r],
                                "c": [c],
                                "v": [v],
                                "w": [w],
                                "var": [var.name],
                                "integral": integral,
                            }
                        )
                        contents["g"] = grouping[s] if s in grouping.keys() else f"{s}"
                        row = pd.DataFrame(contents)
                        df_out = pd.concat([df_out, row], ignore_index=True)
        if df_out.shape[0] == 0:
            return df_out, edges
        bin_names = [n for n in df_out.columns if "bin" in n]
        sumw2_names = [n for n in df_out.columns if "sumw2" in n]
        pdf_names = [n for n in df_out.w.unique() if ("mcreplica" in n)]
        for decor_group, proc_groups in decorrelation_scheme["pdf_mcreplica"].items():
            if len(pdf_names) == 0:
                continue
            for r in regions:
                for c in channels:
                    rms = (
                        df_out.loc[
                            df_out.w.isin(pdf_names)
                            & (df_out.v == "nominal")
                            & (df_out.r == r)
                            & (df_out.c == c),
                            bin_names,
                        ]
                        .std()
                        .values
                    )
                    nom = df_out[
                        (df_out.w == "wgt_nominal")
                        & (df_out.v == "nominal")
                        & (df_out.r == r)
                        & (df_out.c == c)
                    ]
                    nom_bins = nom[bin_names].values[0]
                    nom_sumw2 = nom[sumw2_names].values[0]
                    row_up = {}
                    row_up.update(
                        {f"bin{i}": [nom_bins[i] + rms[i]] for i in range(nbins)}
                    )
                    row_up.update(
                        {f"sumw2_{i}": [nom_sumw2[i]] for i in range(nbins + 1)}
                    )
                    row_up.update({f"sumw2_{i+1}": [sumw2[i]] for i in range(nbins)})
                    row_up.update(
                        {
                            "s": [s],
                            "r": [r],
                            "c": [c],
                            "v": ["nominal"],
                            "w": [f"pdf_mcreplica_{decor_group}_up"],
                            "var": [var.name],
                            "integral": (nom_bins.sum() + rms.sum()),
                        }
                    )
                    row_up["g"] = grouping[s] if s in grouping.keys() else f"{s}"
                    row_down = {}
                    row_down.update(
                        {f"bin{i}": [nom_bins[i] - rms[i]] for i in range(nbins)}
                    )
                    row_down.update(
                        {f"sumw2_{i}": [nom_sumw2[i]] for i in range(nbins + 1)}
                    )
                    row_down.update(
                        {f"sumw2_{i + 1}": [sumw2[i]] for i in range(nbins)}
                    )
                    row_down.update(
                        {
                            "s": [s],
                            "r": [r],
                            "c": [c],
                            "v": ["nominal"],
                            "w": [f"pdf_mcreplica_{decor_group}_down"],
                            "var": [var.name],
                            "integral": (nom_bins.sum() - rms.sum()),
                        }
                    )
                    row_down["g"] = grouping[s] if s in grouping.keys() else f"{s}"
                    df_out = pd.concat(
                        [df_out, pd.DataFrame(row_up), pd.DataFrame(row_down)],
                        ignore_index=True,
                    )
                    df_out = df_out[
                        ~(
                            df_out.w.isin(mcreplicas)
                            & (df_out.v == "nominal")
                            & (df_out.r == r)
                            & (df_out.c == c)
                        )
                    ]

    return df_out, edges


def save_yields(var, hist, edges, args):
    metadata = hist[var.name][["s", "r", "c", "v", "w", "integral"]]
    yields = metadata.groupby(["s", "r", "c", "v", "w"]).aggregate(np.sum)
    yields.to_pickle(f"yields/yields_{args['year']}_{args['label']}.pkl")


def load_yields(s, r, c, v, w, args):
    yields = pd.read_pickle(f"yields/yields_{args['year']}_{args['label']}.pkl")
    filter0 = yields.index.get_level_values("s") == s
    filter1 = yields.index.get_level_values("r") == r
    filter2 = yields.index.get_level_values("c") == c
    filter3 = yields.index.get_level_values("v") == v
    filter4 = yields.index.get_level_values("w") == w
    ret = yields.iloc[filter0 & filter1 & filter2 & filter3 & filter4].values[0][0]
    return ret


def save_shapes(hist, model, args, mva_bins):
    edges = mva_bins[model][args["year"]]
    edges = np.array(edges)
    var = f"score_{model}"
    tdir = f'/depot/cms/hmm/templates/{args["year"]}_{args["label"]}_{var}'
    if args["do_massscan"]:
        mass_point = f'{args["mass"]}'.replace(".", "")
        tdir = tdir.replace("templates", "/templates/massScan/") + f"_{mass_point}"
        try:
            os.mkdir("/depot/cms/hmm/templates/massScan/")
        except Exception:
            pass
    try:
        os.mkdir(tdir)
    except Exception:
        pass

    r_names = {"h-peak": "SR", "h-sidebands": "SB"}

    def get_vwname(v, w):
        vwname = ""
        if "nominal" in v:
            if "off" in w:
                return ""
            elif "nominal" in w:
                vwname = "nominal"
            elif "_up" in w:
                vwname = w.replace("_up", "Up")
            elif "_down" in w:
                vwname = w.replace("_down", "Down")
        else:
            if "nominal" not in w:
                return ""
            elif "_up" in v:
                vwname = v.replace("_up", "Up")
            elif "_down" in v:
                vwname = v.replace("_down", "Down")
            if "jer" not in vwname:
                vwname = "jes" + vwname
        if vwname.startswith("wgt_"):
            vwname = vwname[4:]
        if (
            ("jes" not in vwname)
            and ("btag" not in vwname)
            and ("THU" not in vwname)
            and ("qgl" not in vwname)
            and ("LHE" not in vwname)
        ):
            vwname = vwname.replace("Up", f'{args["year"]}Up').replace(
                "Down", f'{args["year"]}Down'
            )
        return vwname

    try:
        os.mkdir(f'combine/{args["year"]}_{args["label"]}_{var}')
    except Exception:
        pass

    to_variate = [
        "vbf_amcPS",
        "vbf_powhegPS",
        "vbf_powheg_herwig",
        "vbf_powheg_dipole",
        "ewk_lljj_mll105_160_ptj0",
        "ewk_lljj_mll105_160",
        "ewk_lljj_mll105_160_py",
        "ewk_lljj_mll105_160_py_dipole",
    ]
    sample_variations = {
        "SignalPartonShower": {"VBF": ["vbf_powheg_dipole", "vbf_powheg_herwig"]},
        "EWKPartonShower": {
            "EWK": ["ewk_lljj_mll105_160_ptj0", "ewk_lljj_mll105_160_py_dipole"]
        },
    }
    smp_var_shape_only = {"SignalPartonShower": False, "EWKPartonShower": False}

    variated_shapes = {}

    hist = hist[var]
    # centers = (edges[: -1] + edges[1:]) / 2.0
    bin_columns = [c for c in hist.columns if "bin" in c]
    sumw2_columns = [c for c in hist.columns if "sumw2" in c]
    data_names = [n for n in hist.s.unique() if "data" in n]
    for cgroup, cc in args["channel_groups"].items():
        for r in args["regions"]:
            data_obs_hist = np.zeros(len(bin_columns), dtype=float)
            data_obs_sumw2 = np.zeros(len(sumw2_columns), dtype=float)
            for v in hist.v.unique():
                for w in hist.w.unique():
                    vwname = get_vwname(v, w)
                    if vwname == "":
                        continue
                    if ("2016" in args["year"]) and ("pdf_2rms" in vwname):
                        continue
                    if ("2016" not in args["year"]) and ("pdf_mcreplica" in vwname):
                        continue
                    if vwname == "nominal":
                        data_obs = hist[
                            hist.s.isin(data_names) & (hist.r == r) & (hist.c.isin(cc))
                        ]
                        data_obs_hist = data_obs[bin_columns].sum(axis=0).values
                        data_obs_sumw2 = data_obs[sumw2_columns].sum(axis=0).values
                    for c in cc:
                        nom_hist = hist[
                            ~hist.s.isin(data_names)
                            & (hist.v == "nominal")
                            & (hist.w == "wgt_nominal")
                            & (hist.r == r)
                            & (hist.c == c)
                        ]
                        mc_hist = hist[
                            ~hist.s.isin(data_names)
                            & (hist.v == v)
                            & (hist.w == w)
                            & (hist.r == r)
                            & (hist.c == c)
                        ]

                        mc_by_sample = (
                            mc_hist.groupby("s").aggregate(np.sum).reset_index()
                        )
                        for s in mc_by_sample.s.unique():
                            if vwname != "nominal":
                                continue
                            if s in to_variate:
                                variated_shapes[s] = np.array(
                                    mc_hist[mc_hist.s == s][bin_columns].values[0],
                                    dtype=float,
                                )
                        variations_by_group = {}
                        for smp_var_name, smp_var_items in sample_variations.items():
                            if vwname != "nominal":
                                continue
                            if c != "vbf":
                                continue
                            for gr, samples in smp_var_items.items():
                                if len(samples) != 2:
                                    continue
                                if samples[0] not in variated_shapes.keys():
                                    continue
                                if samples[1] not in variated_shapes.keys():
                                    continue
                                variation_up = variated_shapes[samples[0]] - (
                                    variated_shapes[samples[0]]
                                    - variated_shapes[samples[1]]
                                )
                                variation_down = variated_shapes[samples[0]] + (
                                    variated_shapes[samples[0]]
                                    - variated_shapes[samples[1]]
                                )
                                if smp_var_shape_only[smp_var_name]:
                                    histo_nom = np.array(
                                        nom_hist[nom_hist.g == gr][bin_columns].values[
                                            0
                                        ],
                                        dtype=float,
                                    )
                                    norm_up = histo_nom.sum() / variation_up.sum()
                                    norm_down = histo_nom.sum() / variation_down.sum()
                                    variation_up = variation_up * norm_up
                                    variation_down = variation_down * norm_down
                                variations_by_group[gr] = {}
                                variations_by_group[gr][smp_var_name] = [
                                    variation_up,
                                    variation_down,
                                ]

                        mc_hist = mc_hist.groupby("g").aggregate(np.sum).reset_index()
                        for g in mc_hist.g.unique():
                            if g not in grouping.values():
                                continue
                            # if args['do_massscan'] and (g in sig):
                            #     continue
                            decor_ok = True
                            for dec_syst, decorr in decorrelation_scheme.items():
                                if dec_syst in vwname:
                                    for dec_group, proc_groups in decorr.items():
                                        dok = (dec_group in vwname) and (
                                            g not in proc_groups
                                        )
                                        if dok:
                                            decor_ok = False

                            if not decor_ok:
                                continue
                            histo = np.array(
                                mc_hist[mc_hist.g == g][bin_columns].values.sum(axis=0),
                                dtype=float,
                            )
                            if len(histo) == 0:
                                continue
                            sumw2 = np.array(
                                mc_hist[mc_hist.g == g][sumw2_columns].values.sum(
                                    axis=0
                                ),
                                dtype=float,
                            )

                            if sum([sh in w for sh in shape_only]):
                                histo_nom = np.array(
                                    nom_hist[nom_hist.g == g][bin_columns].values.sum(
                                        axis=0
                                    ),
                                    dtype=float,
                                )
                                normalization = histo_nom.sum() / histo.sum()
                                histo = histo * normalization
                                sumw2 = sumw2 * normalization

                            histo[np.isinf(histo)] = 0
                            sumw2[np.isinf(sumw2)] = 0
                            histo[np.isnan(histo)] = 0
                            sumw2[np.isnan(sumw2)] = 0

                            if vwname == "nominal":
                                name = f'{r_names[r]}_{args["year"]}' f"_{g}_{c}"
                            else:
                                name = (
                                    f'{r_names[r]}_{args["year"]}' f"_{g}_{c}_{vwname}"
                                )

                            try:
                                os.mkdir(f"{tdir}/{g}")
                            except Exception:
                                pass
                            np.save(f"{tdir}/{g}/{name}", [histo, sumw2])
                            for groupname, var_items in variations_by_group.items():
                                if (groupname == g) & (vwname == "nominal"):
                                    for variname, variations in var_items.items():
                                        for iud, ud in enumerate(["Up", "Down"]):
                                            if len(variations[iud]) == 0:
                                                variations[iud] = np.ones(len(histo))
                                            # histo_ud =\
                                            #    histo*variations[iud]
                                            histo_ud = variations[iud]
                                            sumw2_ud = np.array(
                                                [0] + list(sumw2[1:] * variations[iud])
                                            )
                                            name = (
                                                f"{r_names[r]}"
                                                f'_{args["year"]}'
                                                f"_{g}_{c}"
                                                f"_{variname}{ud}"
                                            )
                                            np.save(
                                                f"{tdir}/{g}/{name}",
                                                [histo_ud, sumw2_ud],
                                            )

            try:
                os.mkdir(f"{tdir}/Data/")
            except Exception:
                pass
            name = f'{r_names[r]}_{args["year"]}_data_obs'
            np.save(f"{tdir}/Data/{name}", [data_obs_hist, data_obs_sumw2])


def prepare_root_files(var, args, shift_signal=False):
    if "score" in var.name:
        edges = args["mva_bins"][var.name.replace("score_", "")][args["year"]]
    edges = np.array(edges)
    tdir = f"/depot/cms/hmm/templates/" f'{args["year"]}_{args["label"]}_{var.name}'
    if args["do_massscan"]:
        tdir_nominal = tdir
        mass_point = f'{args["mass"]}'.replace(".", "")
        tdir = tdir.replace("templates", "/templates/massScan/") + f"_{mass_point}"
        try:
            os.mkdir(f'combine/massScan_{args["label"]}/')
        except Exception:
            pass
        try:
            os.mkdir(f'combine/massScan_{args["label"]}/{mass_point}')
        except Exception:
            pass
        try:
            os.mkdir(
                f'combine/massScan_{args["label"]}/{mass_point}/'
                f'{args["year"]}_{args["label"]}_{var.name}/'
            )
        except Exception:
            pass
    else:
        try:
            os.mkdir(f'combine/{args["year"]}_{args["label"]}_{var.name}')
        except Exception:
            pass
    regions = ["SB", "SR"]
    centers = (edges[:-1] + edges[1:]) / 2.0

    for cgroup, cc in args["channel_groups"].items():
        for r in regions:
            out_fn = (
                f'combine/{args["year"]}_{args["label"]}'
                f"_{var.name}/shapes_{cgroup}_{r}.root"
            )
            if args["do_massscan"]:
                if shift_signal:
                    out_fn = (
                        f'combine/massScan_{args["label"]}/'
                        f'{mass_point}/{args["year"]}_{args["label"]}'
                        f"_{var.name}/shapes_{cgroup}_{r}.root"
                    )
                else:
                    out_fn = (
                        f'combine/massScan_{args["label"]}'
                        f'/{mass_point}/{args["year"]}_{args["label"]}'
                        f"_{var.name}/shapes_{cgroup}_{r}_nominal.root"
                    )
            out_file = uproot.recreate(out_fn)
            data_hist, data_sumw2 = np.load(
                f'{tdir}/Data/{r}_{args["year"]}_data_obs.npy', allow_pickle=True
            )
            th1_data = from_numpy([data_hist, edges])
            th1_data._fName = "data_obs"
            th1_data._fSumw2 = np.array(data_sumw2)
            th1_data._fTsumw2 = np.array(data_sumw2).sum()
            th1_data._fTsumwx2 = np.array(data_sumw2[1:] * centers).sum()
            out_file["data_obs"] = th1_data
            mc_templates = []
            for group in bkg:
                mc_templates += glob.glob(f"{tdir}/{group}/{r}_*.npy")
            for group in sig:
                if args["do_massscan"] and not shift_signal:
                    mc_templates += glob.glob(f"{tdir_nominal}/{group}/{r}_*.npy")
                else:
                    mc_templates += glob.glob(f"{tdir}/{group}/{r}_*.npy")

            for path in mc_templates:
                if "Data" in path:
                    continue
                if "data_obs" in path:
                    continue
                hist, sumw2 = np.load(path, allow_pickle=True)
                name = os.path.basename(path).replace(".npy", "")
                name = name.replace("DY_filter_vbf_2j", "DYJ2_filter")
                name = name.replace("DY_nofilter_vbf_2j", "DYJ2_nofilter")
                name = name.replace("DY_filter_vbf_01j", "DYJ01_filter")
                name = name.replace("DY_nofilter_vbf_01j", "DYJ01_nofilter")
                name = name.replace("DY_vbf_2j", "DYJ2")
                name = name.replace("DY_vbf_01j", "DYJ01")
                name = name.replace("EWK_vbf", "EWK")
                name = name.replace("TT+ST_vbf", "TT+ST")
                name = name.replace("VV_vbf", "VV")
                name = name.replace("VBF_vbf", "qqH_hmm")
                name = name.replace("ggH_vbf", "ggH_hmm")
                th1 = from_numpy([hist, edges])
                th1._fName = name.replace(f'{r}_{args["year"]}_', "")
                th1._fSumw2 = np.array(sumw2)
                th1._fTsumw2 = np.array(sumw2).sum()
                th1._fTsumwx2 = np.array(sumw2[1:] * centers).sum()
                out_file[name.replace(f'{r}_{args["year"]}_', "")] = th1
            out_file.close()


def get_numbers(var, cc, r, bin_name, args, shift_signal=False):
    groups = {
        "Data": ["vbf"],
        # 'DY_nofilter':['vbf_01j','vbf_2j'],
        # 'DY_filter':['vbf_01j','vbf_2j'],
        "DY": ["vbf_01j", "vbf_2j"],
        "EWK": ["vbf"],
        "TT+ST": ["vbf"],
        "VV": ["vbf"],
        "ggH": ["vbf"],
        "VBF": ["vbf"],
    }
    # regions = ['SB', 'SR']
    year = args["year"]
    # floating_norm = {'DY': ['vbf_01j']}
    sig_groups = ["ggH", "VBF"]
    # sample_variations = {
    #     'SignalPartonShower': {
    #         'VBF': ['vbf_powhegPS', 'vbf_powheg_herwig']},
    #     'EWKPartonShower': {
    #         'EWK': ['ewk_lljj_mll105_160', 'ewk_lljj_mll105_160_py']},
    # }

    sig_counter = 0
    bkg_counter = 0
    tdir = f"/depot/cms/hmm/templates/" f'{args["year"]}_{args["label"]}_{var.name}'
    if args["do_massscan"]:
        tdir_nominal = tdir
        mass_point = f'{args["mass"]}'.replace(".", "")
        tdir = tdir.replace("templates", "/templates/massScan/") + f"_{mass_point}"

    shape_systs_by_group = {}
    for g, cc in groups.items():
        shape_systs_by_group[g] = [
            os.path.basename(path).replace(".npy", "")
            for path in glob.glob(f"{tdir}/{g}/{r}_*.npy")
        ]
        if args["do_massscan"] and (g in sig) and not shift_signal:
            shape_systs_by_group[g] = [
                os.path.basename(path).replace(".npy", "")
                for path in glob.glob(f"{tdir_nominal}/{g}/{r}_*.npy")
            ]
        for c in cc:
            shape_systs_by_group[g] = [
                path
                for path in shape_systs_by_group[g]
                if ((path != f'{r}_{args["year"]}_{g}_{c}') and ("nominal" not in path))
            ]
            shape_systs_by_group[g] = [
                path.replace(f'{r}_{args["year"]}_{g}_{c}_', "")
                .replace("Up", "")
                .replace("Down", "")
                for path in shape_systs_by_group[g]
            ]
        shape_systs_by_group[g] = np.unique(shape_systs_by_group[g])

    shape_systs = []
    for shs in shape_systs_by_group.values():
        shape_systs.extend(shs)
    shape_systs = np.unique(shape_systs)
    shape_systs = [sh for sh in shape_systs if "data_obs" not in sh]

    systs = []
    systs.extend(shape_systs)

    data_yields = pd.DataFrame()
    data_yields["index"] = ["bin", "observation"]

    mc_yields = pd.DataFrame()
    mc_yields["index"] = ["bin", "process", "process", "rate"]

    systematics = pd.DataFrame(index=systs)

    for g, cc in groups.items():
        if g not in grouping.values():
            continue
        for c in cc:
            # gcname = f'{g}_{c}'
            gcname = g
            if g == "VBF":
                gcname = "qqH_hmm"
            if g == "ggH":
                gcname = "ggH_hmm"
            if "01j" in c:
                gcname = gcname.replace("DY", "DYJ01")
            if "2j" in c:
                gcname = gcname.replace("DY", "DYJ2")

            counter = 0
            if g in sig_groups:
                sig_counter += 1
                counter = -sig_counter
            elif "Data" not in g:
                bkg_counter += 1
                counter = bkg_counter
            if g == "Data":
                data = f'{tdir}/Data/{r}_{args["year"]}_data_obs.npy'
                hist, _ = np.load(data, allow_pickle=True)
                rate = hist.sum()
                data_yields.loc[0, "value"] = bin_name
                data_yields.loc[1, "value"] = f"{rate}"
            else:
                nominal_shape = f'{tdir}/{g}/{r}_{args["year"]}_{g}_{c}.npy'
                if (g in sig) and args["do_massscan"] and not shift_signal:
                    nominal_shape = (
                        f"{tdir_nominal}/{g}/" f'{r}_{args["year"]}_{g}_{c}.npy'
                    )
                try:
                    hist, _ = np.load(nominal_shape, allow_pickle=True)
                except Exception:
                    print(f"Can't load templates for: {g} {c}")
                    continue
                rate = hist.sum()
                mc_yields.loc[0, gcname] = bin_name
                mc_yields.loc[1, gcname] = gcname
                mc_yields.loc[2, gcname] = f"{counter}"
                mc_yields.loc[3, gcname] = f"{rate}"
                for syst in shape_systs:
                    systematics.loc[syst, "type"] = "shape"
                    if sum(
                        [dec_syst in syst for dec_syst in decorrelation_scheme.keys()]
                    ):
                        for dec_syst, decorr in decorrelation_scheme.items():
                            if dec_syst in syst:
                                for dec_group, proc_groups in decorr.items():
                                    if dec_group in syst:
                                        if (g in proc_groups) and (
                                            syst in shape_systs_by_group[g]
                                        ):
                                            systematics.loc[syst, gcname] = "1.0"
                                        else:
                                            systematics.loc[syst, gcname] = "-"
                                    # else:
                                    #     systematics.loc[
                                    #         syst,gcname] = '-'
                    # if sum([gname in syst for gname in groups.keys()]):
                    #     systematics.loc[syst,gcname] = '1.0'
                    #         if (g in syst) and
                    #         (syst in shape_systs_by_group[g]) else '-'
                    else:
                        systematics.loc[syst, gcname] = (
                            "1.0" if syst in shape_systs_by_group[g] else "-"
                        )

                # https://twiki.cern.ch/twiki/bin/view/CMS/TWikiLUM
                for year in ["2016", "2017", "2018"]:
                    systematics.loc[f"lumi_uncor{year}", "type"] = "lnN"
                    if args["year"] == year:
                        val = 1.0 + lumi_syst[year]["uncor"] / 100
                        systematics.loc[f"lumi_uncor{year}", gcname] = (
                            "-" if val == 1.0 else f"{val}"
                        )
                    else:
                        systematics.loc[f"lumi_uncor{year}", gcname] = "-"
                for src in lumi_syst[args["year"]].keys():
                    if src == "uncor":
                        continue
                    systematics.loc[f"lumi_{src}", "type"] = "lnN"
                    val = 1.0 + lumi_syst[args["year"]][src] / 100
                    systematics.loc[f"lumi_{src}", gcname] = (
                        "-" if val == 1.0 else f"{val}"
                    )

                for rate_syst, rate_syst_grouping in rate_syst_lookup[year].items():
                    systematics.loc[rate_syst, "type"] = "lnN"
                    if gcname in rate_syst_grouping.keys():
                        val = rate_syst_grouping[gcname]
                    else:
                        val = "-"
                    systematics.loc[rate_syst, gcname] = f"{val}"

    def to_string(df):
        string = ""
        for row in df.values:
            for i, item in enumerate(row):
                ncols = 2 if item in ["bin", "process", "rate", "observation"] else 1
                # print(item)
                row[i] = item + " " * (ncols * 20 - len(item))
            string += " ".join(row)
            string += "\n"
        return string

    print(data_yields)
    print(mc_yields)
    print(systematics)
    return (
        to_string(data_yields),
        to_string(mc_yields),
        to_string(systematics.reset_index()),
    )


def make_datacards(var, args, shift_signal=False):
    year = args["year"]
    for cgroup, cc in args["channel_groups"].items():
        for r in ["SB", "SR"]:
            bin_name = f"{cgroup}_{r}_{year}"
            datacard_name = (
                f'combine/{year}_{args["label"]}_{var.name}'
                f"/datacard_{cgroup}_{r}.txt"
            )
            if args["do_massscan"]:
                mass_point = f'{args["mass"]}'.replace(".", "")
                if shift_signal:
                    datacard_name = (
                        f'combine/massScan_{args["label"]}/'
                        f'{mass_point}/{args["year"]}_{args["label"]}'
                        f"_{var.name}/datacard_{cgroup}_{r}.txt"
                    )
                else:
                    datacard_name = (
                        f'combine/massScan_{args["label"]}/'
                        f'{mass_point}/{args["year"]}_{args["label"]}'
                        f"_{var.name}/datacard_{cgroup}_{r}_nominal.txt"
                    )
            if args["do_massscan"] and shift_signal:
                shapes_file = f"shapes_{cgroup}_{r}.root"
            else:
                shapes_file = f"shapes_{cgroup}_{r}_nominal.root"
            datacard = open(datacard_name, "w")
            datacard.write("imax 1\n")
            datacard.write("jmax *\n")
            datacard.write("kmax *\n")
            datacard.write("---------------\n")
            datacard.write(
                f"shapes * {bin_name} {shapes_file} " "$PROCESS $PROCESS_$SYSTEMATIC\n"
            )
            datacard.write("---------------\n")
            data_yields, mc_yields, systematics = get_numbers(
                var, cc, r, bin_name, args, shift_signal
            )
            datacard.write(data_yields)
            datacard.write("---------------\n")
            datacard.write(mc_yields)
            datacard.write("---------------\n")
            datacard.write(systematics)
            datacard.write(
                f"XSecAndNorm{year}DYJ01  rateParam " f"{bin_name} DYJ01 1 [0.2,5] \n"
            )
            datacard.write(f"{bin_name} autoMCStats 0 1 1\n")
            datacard.write("---------------\n")
            datacard.write(
                "nuisance edit rename"
                " (DYJ2|DYJ01|ggH_hmm|TT+ST|VV) * "
                "qgl_wgt  QGLweightPY \n"
            )
            datacard.write("nuisance edit rename EWK * qgl_wgt" " QGLweightHER \n")
            datacard.write(
                "nuisance edit rename qqH_hmm * qgl_wgt" " QGLweightPYDIPOLE \n"
            )
            datacard.close()
            print(f"Saved datacard to {datacard_name}")
    return


def add_source(hist, group_name):
    bin_columns = [c for c in hist.columns if "bin" in c]
    sumw2_columns = [c for c in hist.columns if "sumw2" in c]
    vals = hist[hist["g"] == group_name]
    vals = vals.groupby("g").aggregate(np.sum).reset_index()
    sumw2 = vals[sumw2_columns].sum(axis=0).reset_index(drop=True).values
    try:
        vals = vals[bin_columns].values.sum(axis=0)
        return vals, sumw2
    except Exception:
        return np.array([]), np.array([])


def plot(
    var,
    hists,
    edges,
    args,
    r="",
    save=True,
    blind=True,
    show=False,
    plotsize=12,
    compare_with_pisa=False,
):
    hist = hists[var.name]
    edges_data = edges
    blind_bins = 5
    if r == "h-sidebands":
        blind = False

    if r != "":
        hist = hist[hist.r == r]
    year = args["year"]
    label = args["label"]
    bin_columns = [c for c in hist.columns if "bin" in c]
    # sumw2_columns = [c for c in hist.columns if 'sumw2' in c]

    def get_shapes_for_option(hist_, v, w):
        bkg_groups = ["DY", "DY_nofilter", "DY_filter", "EWK", "TT+ST", "VV"]
        hist_nominal = hist_[(hist_.w == "wgt_nominal") & (hist_.v == "nominal")]
        hist = hist_[(hist_.w == w) & (hist_.v == v)]
        vbf, vbf_sumw2 = add_source(hist, "VBF")
        ggh, ggh_sumw2 = add_source(hist, "ggH")
        # ewk, ewk_sumw2 = add_source(hist, 'EWK')
        # ewk_py, ewk_py_sumw2 = add_source(hist, 'EWK_Pythia')
        data, data_sumw2 = add_source(hist_nominal, "Data")

        bin_columns = [c for c in hist.columns if "bin" in c]
        sumw2_columns = [c for c in hist.columns if "sumw2" in c]
        bkg_df = hist[hist["g"].isin(bkg_groups)]
        bkg_df.loc[bkg_df.g.isin(["DY_nofilter", "DY_filter"]), "g"] = "DY"
        bkg_df.loc[:, "bkg_integral"] = bkg_df[bin_columns].sum(axis=1)
        bkg_df = bkg_df.groupby("g").aggregate(np.sum).reset_index()
        bkg_df = bkg_df.sort_values(by="bkg_integral").reset_index(drop=True)
        bkg_total = bkg_df[bin_columns].sum(axis=0).reset_index(drop=True)
        bkg_sumw2 = bkg_df[sumw2_columns].sum(axis=0).reset_index(drop=True)

        return {
            "data": data,
            "data_sumw2": data_sumw2,
            "vbf": vbf,
            "vbf_sumw2": vbf_sumw2,
            "ggh": ggh,
            "ggh_sumw2": ggh_sumw2,
            # 'ewk':        ewk,
            # 'ewk_sumw2': ewk_sumw2,
            # 'ewk_py':        ewk_py,
            # 'ewk_py_sumw2': ewk_py_sumw2,
            "bkg_df": bkg_df,
            "bkg_total": bkg_total,
            "bkg_sumw2": bkg_sumw2,
        }

    ret_nominal = get_shapes_for_option(hist, "nominal", "wgt_nominal")
    # ret_nnlops_off = get_shapes_for_option(
    #     hist, 'nominal', 'wgt_nnlops_off')
    data = ret_nominal["data"]
    data_sumw2 = ret_nominal["data_sumw2"][1:]

    if blind and "score" in var.name:
        data = data[:-blind_bins]
        data_sumw2 = data_sumw2[:-blind_bins]
        edges_data = edges_data[:-blind_bins]

    vbf = ret_nominal["vbf"]
    # vbf_sumw2 = ret_nominal['vbf_sumw2'][1:]
    ggh = ret_nominal["ggh"]
    # ggh_sumw2 = ret_nominal['ggh_sumw2'][1:]

    # ggh_nnlops_off = ret_nnlops_off['ggh']
    # ggh_sumw2_nnlops_off = ret_nnlops_off['ggh_sumw2'][1:]

    # ewk        = ret_nominal['ewk']
    # ewk_sumw2  = ret_nominal['ewk_sumw2'][1:]
    # ewk_py        = ret_nominal['ewk_py']
    # ewk_py_sumw2  = ret_nominal['ewk_py_sumw2'][1:]

    bkg_df = ret_nominal["bkg_df"]
    bkg_total = ret_nominal["bkg_total"]
    bkg_sumw2 = ret_nominal["bkg_sumw2"][1:].values

    if len(bkg_df.values) > 1:
        bkg = np.stack(bkg_df[bin_columns].values)
        stack = True
    else:
        bkg = bkg_df[bin_columns].values
        stack = False
    bkg_labels = bkg_df.g

    # Report yields
    if not show and var.name == "dimuon_mass":
        print("=" * 50)
        if r == "":
            print(f"{var.name}: Inclusive yields:")
        else:
            print(f"{var.name}: Yields in {r}")
        print("=" * 50)
        print("Data", data.sum())
        for row in bkg_df[["g", "integral"]].values:
            print(row)
        print("VBF", vbf.sum())
        print("ggH", ggh.sum())
        print("-" * 50)

    # Make plot
    fig = plt.figure()
    plt.rcParams.update({"font.size": 22})
    ratio_plot_size = 0.25
    data_opts = {"color": "k", "marker": ".", "markersize": 15}
    data_opts_pisa = {"color": "red", "marker": ".", "markersize": 15}
    stack_fill_opts = {"alpha": 0.8, "edgecolor": (0, 0, 0)}
    stat_err_opts = {
        "step": "post",
        "label": "Stat. unc.",
        "hatch": "//////",
        "facecolor": "none",
        "edgecolor": (0, 0, 0, 0.5),
        "linewidth": 0,
    }
    ratio_err_opts = {"step": "post", "facecolor": (0, 0, 0, 0.3), "linewidth": 0}

    fig.clf()
    fig.set_size_inches(plotsize, plotsize * (1 + ratio_plot_size))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[(1 - ratio_plot_size), ratio_plot_size], hspace=0.05
    )

    # Top panel: Data/MC
    plt1 = fig.add_subplot(gs[0])

    if bkg_total.sum():
        ax_bkg = hep.histplot(
            bkg,
            edges,
            ax=plt1,
            label=bkg_labels,
            stack=stack,
            histtype="fill",
            **stack_fill_opts,
        )
        err = coffea.hist.plot.poisson_interval(np.array(bkg_total), bkg_sumw2)
        ax_bkg.fill_between(
            x=edges,
            y1=np.r_[err[0, :], err[0, -1]],
            y2=np.r_[err[1, :], err[1, -1]],
            **stat_err_opts,
        )

    if compare_with_pisa:
        r_opt = "inclusive" if r == "" else r
        pisa_hist, pisa_data_hist = get_pisa_hist(var, r_opt, edges)
        if pisa_hist.sum():
            hep.histplot(
                pisa_hist,
                edges,
                label="Pisa",
                histtype="step",
                **{"linewidth": 3, "color": "red"},
            )
        if pisa_data_hist.sum():
            hep.histplot(
                pisa_data_hist,
                edges,
                label="Pisa Data",
                histtype="errorbar",
                yerr=np.sqrt(pisa_data_hist),
                **data_opts_pisa,
            )

    # if ewk.sum():
    #     ax_ewk = hep.histplot(ewk, edges, label='EWK MG',
    #                           histtype='step', **{'linewidth':3,
    #                                               'color':'red'})
    # if ewk_py.sum():
    #     ax_ewk_py = hep.histplot(ewk_py, edges, label='EWK Pythia',
    #                              histtype='step', **{'linewidth':3,
    #                                                  'color':'blue'})

    if ggh.sum():
        hep.histplot(
            ggh,
            edges,
            label="ggH",
            histtype="step",
            **{"linewidth": 3, "color": "lime"},
        )
    # if ggh_nnlops_off.sum():
    #     ax_ggh = hep.histplot(ggh_nnlops_off, edges,
    #                           label='ggH NNLOPS off', histtype='step',
    #                           **{'linewidth':3, 'color':'violet'})

    if vbf.sum():
        hep.histplot(
            vbf,
            edges,
            label="VBF",
            histtype="step",
            **{"linewidth": 3, "color": "aqua"},
        )
    if data.sum():
        hep.histplot(
            data,
            edges_data,
            label="Data",
            histtype="errorbar",
            yerr=np.sqrt(data),
            **data_opts,
        )

    max_variation_up = bkg_total.sum()
    max_variation_down = bkg_total.sum()
    # max_var_up_name = ''
    # max_var_down_name = ''
    for v in hist.v.unique():
        for w in hist.w.unique():
            continue
            if ("nominal" in v) and ("nominal" in w):
                continue
            if "off" in w:
                continue
            if "wgt" not in w:
                continue
            # if ('jer1' not in v): continue
            # if ('EC2' not in v) or ('EC22016' in v): continue
            if "HF2016" not in v:
                continue
            if "nominal" not in w:
                continue
            ret = get_shapes_for_option(hist, v, w)
            if ret["bkg_total"].sum():
                hep.histplot(
                    ret["bkg_total"].values, edges, histtype="step", **{"linewidth": 3}
                )
                if ret["bkg_total"].values.sum() > max_variation_up:
                    max_variation_up = ret["bkg_total"].values.sum()
                    # max_var_up_name = f'{v},{w}'
                if ret["bkg_total"].values.sum() < max_variation_down:
                    max_variation_down = ret["bkg_total"].values.sum()
                    # max_var_down_name = f'{v},{w}'

            if ret["ggh"].sum():
                hep.histplot(ret["ggh"], edges, histtype="step", **{"linewidth": 3})
            if ret["vbf"].sum():
                hep.histplot(ret["vbf"], edges, histtype="step", **{"linewidth": 3})

    lbl = hep.cms.label(ax=plt1, data=True, paper=False, year=year)

    plt1.set_yscale("log")
    plt1.set_ylim(0.01, 1e9)
    # plt1.set_xlim(var.xmin,var.xmax)
    plt1.set_xlim(edges[0], edges[-1])
    plt1.set_xlabel("")
    plt1.tick_params(axis="x", labelbottom=False)
    plt1.legend(prop={"size": "small"})

    # Bottom panel: Data/MC ratio plot
    plt2 = fig.add_subplot(gs[1], sharex=plt1)

    cond = (data.sum() * bkg_total.sum()) and (not blind or var.name != "dimuon_mass")
    if cond:
        ratios = np.zeros(len(data))
        yerr = np.zeros(len(data))
        unity = np.ones_like(bkg_total)
        bkg_total[bkg_total == 0] = 1e-20
        ggh[ggh == 0] = 1e-20
        vbf[vbf == 0] = 1e-20
        bkg_unc = coffea.hist.plot.poisson_interval(unity, bkg_sumw2 / bkg_total**2)
        denom_unc = bkg_unc

        if blind and "score" in var.name:
            bkg_total_ = bkg_total[:-blind_bins]
        else:
            bkg_total_ = bkg_total
        mask = bkg_total_ != 0
        ratios[mask] = np.array(data[mask] / bkg_total_[mask])
        yerr[mask] = np.sqrt(data[mask]) / bkg_total_[mask]
        edges_ratio = edges_data if blind else edges
        ax_ratio = hep.histplot(
            ratios, edges_ratio, histtype="errorbar", yerr=yerr, **data_opts
        )
        ax_ratio.fill_between(
            edges,
            np.r_[denom_unc[0], denom_unc[0, -1]],
            np.r_[denom_unc[1], denom_unc[1, -1]],
            label="Stat. unc.",
            **ratio_err_opts,
        )

    for v in hist.v.unique():
        for w in hist.w.unique():
            # continue
            if ("nominal" not in v) and ("nominal" not in w):
                continue
            if ("nominal" in v) and ("nominal" in w):
                continue
            if "off" in w:
                continue
            # if ('qgl' not in w): continue
            if "LHERen_DY" not in w:
                continue
            # if ('RelativeSample2018' not in v): continue
            # if ('EC2' not in v) or ('EC22016' in v): continue
            # if ('nominal' not in w): continue
            print(f"Add plot for {v} {w}")
            ret = get_shapes_for_option(hist, v, w)
            syst_ratio = np.zeros(len(bkg_total))
            lbl = f"{v}" if w == "nominal" else f"{w}"
            syst_ratio[bkg_total != 0] = np.array(
                ret["bkg_total"].values[bkg_total != 0] / bkg_total[bkg_total != 0]
            )
            hep.histplot(
                syst_ratio, edges, histtype="step", label=lbl, **{"linewidth": 3}
            )
            plt2.legend(prop={"size": "xx-small"})

    plot_all_systematics = False
    if plot_all_systematics:
        total_err2_up = np.zeros(len(bkg_total))
        total_err2_down = np.zeros(len(bkg_total))
        ratio_up = np.ones(len(bkg_total))
        ratio_down = np.ones(len(bkg_total))
        mask = bkg_total != 0
        for v in hist.v.unique():
            for w in hist.w.unique():
                if ("nominal" not in v) and ("nominal" not in w):
                    continue
                if ("nominal" in v) and ("nominal" in w):
                    continue
                if "off" in w:
                    continue
                ret = get_shapes_for_option(hist, v, w)
                if sum(ret["bkg_total"].values[mask]) != 0:
                    if ("_up" in w) or ("_up" in v):
                        total_err2_up += np.square(
                            (ret["bkg_total"].values[mask] - bkg_total[mask])
                            / bkg_total[mask]
                        )
                    if ("_down" in w) or ("_down" in v):
                        total_err2_down += np.square(
                            (ret["bkg_total"].values[mask] - bkg_total[mask])
                            / bkg_total[mask]
                        )
        ratio_up[mask] = 1 + np.sqrt(total_err2_up)[mask]
        ratio_down[mask] = 1 - np.sqrt(total_err2_down)[mask]
        hep.histplot(
            ratio_up,
            edges,
            histtype="step",
            label="Total syst. unc.",
            **{"linewidth": 3, "color": "red"},
        )
        hep.histplot(
            ratio_down, edges, histtype="step", **{"linewidth": 3, "color": "red"}
        )
        plt2.legend(prop={"size": "xx-small"})

    plt2.axhline(1, ls="--")
    plt2.set_ylim([0.5, 1.5])
    plt2.set_ylabel("Data/MC")
    lbl = plt2.get_xlabel()
    plt2.set_xlabel(f"{var.caption}")

    if compare_with_pisa and pisa_hist.sum():
        ratio = np.zeros(len(bkg_total))
        ratio[bkg_total != 0] = np.array(
            pisa_hist[bkg_total != 0] / bkg_total[bkg_total != 0]
        )
        hep.histplot(
            ratio,
            edges,
            label="Pisa/Purdue MC",
            histtype="step",
            **{"linewidth": 3, "color": "red"},
        )
        plt2.legend(prop={"size": "small"})
        plt2.set_ylim([0.8, 1.2])

    if compare_with_pisa and pisa_data_hist.sum():
        ratio_data = np.zeros(len(bkg_total))
        ratio_data[data != 0] = np.array(pisa_data_hist[data != 0] / data[data != 0])
        hep.histplot(
            ratio_data,
            edges,
            label="Pisa/Purdue Data",
            histtype="step",
            **{"linewidth": 3, "color": "blue"},
        )
        plt2.legend(prop={"size": "small"})
        plt2.set_ylim([0.8, 1.2])

    if save:
        # Save plots
        out_path = args["out_path"]
        full_out_path = f"{out_path}/plots_{year}_{label}"
        if "plot_extra" in args.keys():
            if args["plot_extra"]:
                full_out_path += "_extra"
        try:
            os.mkdir(out_path)
        except Exception:
            pass
        try:
            os.mkdir(full_out_path)
        except Exception:
            pass
        if r == "":
            out_name = f"{full_out_path}/{var.name}_inclusive.png"
        else:
            out_name = f"{full_out_path}/{var.name}_{r}.png"
        print(f"Saving {out_name}")
        fig.savefig(out_name)

    if show:
        plt.show()


var_map_pisa = {
    "mu1_pt": "Mu0_pt",
    "mu1_eta": "Mu0_eta",
    "mu2_pt": "Mu1_pt",
    "mu2_eta": "Mu1_eta",
    "dimuon_pt": "Higgs_pt",
    "dimuon_eta": "Higgs_eta",
    "dimuon_mass": "Higgs_m",
    "jet1_pt": "QJet0_pt_touse",
    "jet1_phi": "QJet0_phi",
    "jet1_eta": "QJet0_eta",
    "jet2_pt": "QJet1_pt_touse",
    "jet2_phi": "QJet1_phi",
    "jet2_eta": "QJet1_eta",
}


def get_pisa_hist(var, r, edges):
    import uproot

    filenames = {
        "data": "/depot/cms/hmm/coffea/pisa-jun12/data2018Snapshot.root",
        "dy_m105_160_amc": "/depot/cms/hmm/coffea/pisa/DY105_2018AMCPYSnapshot.root",
        "dy_m105_160_vbf_amc": "/depot/cms/hmm/coffea/pisa/DY105VBF_2018AMCPYSnapshot.root",
    }
    xsec = {"data": 1, "dy_m105_160_amc": 47.17, "dy_m105_160_vbf_amc": 2.03}
    N = {
        "data": 1,
        "dy_m105_160_amc": 6995355211.029654,
        "dy_m105_160_vbf_amc": 3146552884.4507833,
    }
    qgl_mean = {
        "data": 1,
        "dy_m105_160_amc": 1.04859375342,
        "dy_m105_160_vbf_amc": 1.00809412662,
    }

    lumi = 59970.0
    # samples = ['dy_m105_160_amc', 'dy_m105_160_vbf_amc']
    samples = ["data"]
    total_hist = np.array([])
    data_hist = np.array([])
    for s in samples:
        with uproot.open(filenames[s]) as f:
            tree = f["Events"]
            wgt = np.ones(len(tree["event"].array()), dtype=float)
            weights = [
                "genWeight",
                "puWeight",
                "btagEventWeight",
                "muEffWeight",
                "EWKreweight",
                "PrefiringWeight",
                "QGLweight",
            ]
            if "data" not in s:
                for i, w in enumerate(weights):
                    wgt = wgt * tree[w].array()
                wgt = wgt * xsec[s] * lumi / N[s] / qgl_mean[s]
            hist = bh.Histogram(bh.axis.Variable(edges))
            var_arr = tree[var_map_pisa[var.name]].array()
            hist.fill(var_arr, weight=wgt)
            if "data" in s:
                data_hist = hist.to_numpy()[0]
                continue
            if len(total_hist) > 0:
                total_hist += hist.to_numpy()[0]
            else:
                total_hist = hist.to_numpy()[0]

    print(f"Pisa data yield ({var.name}):", data_hist.sum())
    return total_hist, data_hist
