import numpy as np
import pickle
import torch
from stage2.mva_models import Net, NetPisaRun2, NetPisaRun2Combination


training_features = [
    "dimuon_mass",
    "dimuon_pt",
    "dimuon_pt_log",
    "dimuon_eta",
    # "dimuon_ebe_mass_res",
    # "dimuon_ebe_mass_res_rel",
    # "dimuon_cos_theta_cs",
    # "dimuon_phi_cs",
    "dimuon_pisa_mass_res",
    "dimuon_pisa_mass_res_rel",
    "dimuon_cos_theta_cs_pisa",
    "dimuon_phi_cs_pisa",
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


training_features_mass = [
    "dimuon_mass",
    "dimuon_pisa_mass_res",
    "dimuon_pisa_mass_res_rel",
]

training_features_nomass = [
    "dimuon_pt",
    "dimuon_pt_log",
    "dimuon_eta",
    "dimuon_cos_theta_cs_pisa",
    "dimuon_phi_cs_pisa",
    "jet1_pt_nominal",
    "jet1_eta_nominal",
    "jet1_phi_nominal",
    "jet1_qgl_nominal",
    "jet2_pt_nominal",
    "jet2_eta_nominal",
    "jet2_phi_nominal",
    "jet2_qgl_nominal",
    "jj_mass_nominal",
    "jj_mass_log_nominal",
    "jj_dEta_nominal",
    "rpt_nominal",
    "ll_zstar_log_nominal",
    "mmj_min_dEta_nominal",
    "nsoftjets5_nominal",
    "htsoft2_nominal",
    "year",
]


def prepare_features(df, parameters, variation="nominal", add_year=False):
    global training_features
    if add_year:
        features = training_features + ["year"]
    else:
        features = training_features
    features_var = []
    for trf in features:
        if f"{trf}_{variation}" in df.columns:
            features_var.append(f"{trf}_{variation}")
        elif trf in df.columns:
            features_var.append(trf)
        else:
            print(f"Variable {trf} not found in training dataframe!")
    return features_var


def evaluate_pytorch_dnn(df, variation, model, parameters, score_name, channel):
    features = prepare_features(df, parameters, variation, add_year=True)

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
        model_path = f"{parameters['models_path']}/{channel}/models/{model}_{i}.pt"
        dnn_model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        dnn_model.eval()

        """
        output = dnn_model.pre_output(df_i).detach().numpy()
        import pandas as pd
        print(pd.DataFrame(output))
        print(pd.DataFrame(output).value_counts())
        print(pd.DataFrame(output).value_counts().values.max())
        #print(dnn_model(df_i).detach().numpy()[0] - 0.9173)
        import sys
        sys.exit()
        """
        df.loc[eval_filter, score_name] = np.arctanh((dnn_model(df_i).detach().numpy()))

    return df[score_name]


def evaluate_pytorch_dnn_pisa(
    df, variation, model_name, parameters, score_name, channel
):
    features = prepare_features(df, parameters, variation, add_year=True)

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

        scalers_path = f"{parameters['models_path']}/{channel}/scalers/scalers_{model_name}_{i}.npy"
        scalers = np.load(scalers_path)
        df_i = df.loc[eval_filter, :]
        if df_i.shape[0] == 0:
            continue

        df_i.loc[df_i.region != "h-peak", "dimuon_mass"] = 125.0

        df_i = (df_i[features] - scalers[0]) / scalers[1]
        # df_i = torch.tensor(df_i.values).float()
        df_i_mass = df_i[training_features_mass]
        df_i_nomass = df_i[training_features_nomass]
        df_i_mass = torch.tensor(df_i_mass.values).float()
        df_i_nomass = torch.tensor(df_i_nomass.values).float()

        nlayers = 3
        nnodes = [64, 32, 16]
        freeze = []

        training_setup = {
            "sig_vs_ewk": {
                "datasets": [
                    "ewk_lljj_mll105_160_py_dipole",
                    "ggh_amcPS",
                    "vbf_powheg_dipole",
                ],
                "features": training_features_mass + training_features_nomass,
            },
            "sig_vs_dy": {
                "datasets": [
                    "dy_m105_160_amc",
                    "dy_m105_160_vbf_amc",
                    "ggh_amcPS",
                    "vbf_powheg_dipole",
                ],
                "features": training_features_mass + training_features_nomass,
            },
            "no_mass": {
                "datasets": [
                    "dy_m105_160_amc",
                    "dy_m105_160_vbf_amc",
                    "ttjets_dl",
                    "ggh_amcPS",
                    "vbf_powheg_dipole",
                    "ewk_lljj_mll105_160_py_dipole",
                ],
                "features": training_features_nomass,
            },
            "mass": {
                "datasets": [
                    "dy_m105_160_amc",
                    "dy_m105_160_vbf_amc",
                    "ttjets_dl",
                    "ggh_amcPS",
                    "vbf_powheg_dipole",
                    "ewk_lljj_mll105_160_py_dipole",
                ],
                "features": training_features_mass,
            },
            "combination": {
                "datasets": [
                    "dy_m105_160_amc",
                    "dy_m105_160_vbf_amc",
                    "ttjets_dl",
                    "ggh_amcPS",
                    "vbf_powheg_dipole",
                    "ewk_lljj_mll105_160_py_dipole",
                ],
            },
        }
        subnetworks = {}
        for name in ["sig_vs_ewk", "sig_vs_dy", "no_mass", "mass"]:
            subnetworks[name] = NetPisaRun2(
                name, len(training_setup[name]["features"]), nlayers, nnodes
            )
            # subnetworks[name].to(device)
            model_path = f"data/trained_models/vbf/models/{model_name}_{name}_{i}.pt"
            subnetworks[name].load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
        dnn_model = NetPisaRun2Combination(
            "combination", nlayers, nnodes, subnetworks, freeze
        )

        model_path = f"{parameters['models_path']}/{channel}/models/{model_name}_combination_{i}.pt"
        dnn_model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        dnn_model.eval()
        df.loc[eval_filter, score_name] = np.arctanh(
            (dnn_model(df_i_nomass, df_i_mass).detach().numpy())
        )

    return df[score_name]


def evaluate_bdt(df, variation, model, parameters, score_name):
    # if parameters["do_massscan"]:
    #     mass_shift = parameters["mass"] - 125.0
    features = prepare_features(df, parameters, variation, add_year=False)
    score_name = f"score_{model}_{variation}"
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
