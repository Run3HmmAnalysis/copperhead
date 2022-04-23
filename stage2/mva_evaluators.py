import numpy as np
import pickle
import torch
from stage2.mva_models import Net


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


def prepare_features(df, parameters, variation="nominal", add_year=False):
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


def evaluate_pytorch_dnn(df, variation, model, parameters, score_name, channel):
    features = prepare_features(df, parameters, variation)
    try:
        df = df.compute()
    except Exception:
        pass

    if df.shape[0] == 0:
        return None

    df.loc[:, score_name] = 0

    nfolds = 4
    for i in range(nfolds):
        # train_folds = [(i + f) % nfolds for f in [0, 1]]
        # val_folds = [(i + f) % nfolds for f in [2]]
        eval_folds = [(i + f) % nfolds for f in [3]]

        eval_filter = df.event.mod(nfolds).isin(eval_folds)

        scalers_path = (
            f"{parameters['models_path']}/{channel}/scalers/scalers_{model}_{i}.npy"
        )
        scalers = np.load(scalers_path)
        df_i = df.loc[eval_filter, :]
        if df_i.shape[0] == 0:
            continue
        df_i.loc[df_i.region != "h-peak", "dimuon_mass"] = 125.0
        df_i = (df_i[features] - scalers[0]) / scalers[1]
        df_i = torch.tensor(df_i.values).float()

        dnn_model = Net()
        model_path = (
            f"{parameters['models_path']}/{channel}/models/model_{model}_{i}.pt"
        )
        dnn_model.load_state_dict(torch.load(model_path))
        dnn_model.eval()
        df.loc[eval_filter, score_name] = np.arctanh((dnn_model(df_i).detach().numpy()))

    return df[score_name]


def evaluate_bdt(df, variation, model, parameters, score_name):
    # if parameters["do_massscan"]:
    #     mass_shift = parameters["mass"] - 125.0
    features = prepare_features(df, parameters, variation, add_year=False)
    score_name = f"score_{model} {variation}"
    try:
        df = df.compute()
    except Exception:
        pass

    if df.shape[0] == 0:
        return None

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
            f"{parameters['models_path']}/{model}/BDT_model_earlystop50_{label}.pkl"
        )

        bdt_model = pickle.load(open(model_path, "rb"))
        df_i = df[eval_filter]
        if df_i.shape[0] == 0:
            continue
        df_i.loc[df_i.region != "h-peak", "dimuon_mass"] = 125.0
        # if parameters["do_massscan"]:
        #     df_i.loc[:, "dimuon_mass"] = df_i["dimuon_mass"] - mass_shift
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
