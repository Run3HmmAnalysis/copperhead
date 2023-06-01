import glob
import tqdm
import copy
from stage2.categorizer import split_into_channels, categorize_dnn_output
from stage2.mva_evaluators import evaluate_pytorch_dnn
from python.io import load_dataframe
import pandas as pd
from stage2.mva_models import Net

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


training_features = [
    "dimuon_mass",
    "dimuon_pt",
    "dimuon_pt_log",
    "dimuon_eta",
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
    "year",
]

parameters = {
    "global_path": "/depot/cms/hmm/copperhead/",
    "label": "2022jul29",
    "channels": ["vbf"],
    "custom_npartitions": {"vbf_powheg_dipole": 1, "ggh_amcPS": 1},
    "models_path": "data/trained_models/",
}

datasets = [
    "vbf_powheg_dipole",
    "ggh_amcPS",
    "dy_m105_160_amc",
    "dy_m105_160_vbf_amc",
    "ewk_lljj_mll105_160_ptj0",
]
model_name = "pytorch_jul12"
score_name = f"score_{model_name}_nominal"
channel = "vbf"
region = "h-peak"

signals = ["vbf_powheg_dipole", "ggh_amcPS"]
backgrounds = ["dy_m105_160_amc", "dy_m105_160_vbf_amc", "ewk_lljj_mll105_160_ptj0"]


def get_mean_grads(df, features, scalers, dnn_model, criterion):
    inputs = df[features]
    inputs = (inputs - scalers[0]) / scalers[1]
    inputs = torch.tensor(inputs.values.astype(float)).float()
    inputs.requires_grad = True

    output = dnn_model(inputs)
    loss = criterion(
        output,
        torch.tensor([1 for ii in range(df.shape[0])]).view(df.shape[0], 1).float(),
    )
    loss.backward()

    grads = inputs.grad.numpy()
    grads = pd.DataFrame(abs(grads), columns=training_features)
    return grads.mean().sort_values(ascending=False)


def get_importance(df, features, feature, scalers, dnn_model, criterion):
    inputs = copy.deepcopy(df[features])
    inputs = (inputs - scalers[0]) / scalers[1]
    if feature in features:
        if f"{feature}_log" in features:
            feat_log = f"{feature}_log"
            inputs[[feature, feat_log]] = (
                inputs[[feature, feat_log]].sample(frac=1).values
            )
        elif ("_log" in feature) and (feature.replace("_log", "") in features):
            feat_log = feature.replace("_log", "")
            inputs[[feature, feat_log]] = (
                inputs[[feature, feat_log]].sample(frac=1).values
            )
        elif feature.replace("_nominal", "_log_nominal") in features:
            feat_log = feature.replace("_nominal", "_log_nominal")
            inputs[[feature, feat_log]] = (
                inputs[[feature, feat_log]].sample(frac=1).values
            )
        elif ("_log" in feature) and (
            feature.replace("_log_nominal", "_nominal") in features
        ):
            feat_log = feature.replace("_log_nominal", "_nominal")
            inputs[[feature, feat_log]] = (
                inputs[[feature, feat_log]].sample(frac=1).values
            )
        else:
            inputs[feature] = inputs[feature].sample(frac=1).values

    inputs = torch.tensor(inputs.values.astype(float)).float()
    # inputs.requires_grad = True

    df[f"dnn_score_{feature}_shuffled"] = dnn_model(inputs).detach().numpy()
    scores = [feature, f"dnn_score_{feature}_shuffled"]
    to_return = []

    for score in scores:
        if score not in df.columns:
            to_return.append(0)
            continue
        nbins = 20
        grid = [(i + 1) / nbins for i in range(nbins)]
        bins = np.quantile(df.loc[df.dataset.isin(signals), score].sort_values(), grid)

        df.loc[:, "bin_number"] = 0
        for i in range(0, nbins - 1):
            lo = bins[i]
            hi = bins[i + 1]
            cut = (df[score] > lo) & (df[score] <= hi)
            df.loc[cut, "bin_number"] = i

        s = df.loc[df.dataset.isin(signals)].groupby("bin_number")["wgt_nominal"].sum()
        b = (
            df.loc[df.dataset.isin(backgrounds)]
            .groupby("bin_number")["wgt_nominal"]
            .sum()
        )

        # significance2 = (s*s / (s+b)).sum()

        significance2 = 2 * ((s + b) * np.log(1 + s / b) - s).sum()

        to_return.append(significance2)

    # print(feature, to_return)
    return to_return


df_path = "/depot/cms/hmm/coffea/training_dataset_jun9_vbf.pickle"
df = pd.read_pickle(df_path)

wgts = [c for c in df.columns if "wgt" in c]
df.loc[:, wgts] = df.loc[:, wgts].fillna(0)
df.fillna(-999.0, inplace=True)
# for c in df.columns:
#    df.loc[df[c]==-99.0, c] = -999.0
df = df[~((df.dataset == "dy_m105_160_amc") & (df.gjj_mass > 350))]
df = df[~((df.dataset == "dy_m105_160_vbf_amc") & (df.gjj_mass <= 350))]
df = df[df.region == "h-peak"]
# print(df.groupby("dataset")["wgt_nominal"].sum())

features = []
feat_map = {"total": "total"}
for f in training_features:
    if f in df.columns:
        features.append(f)
        feat_map[f] = f
    elif f"{f}_nominal" in df.columns:
        features.append(f"{f}_nominal")
        feat_map[f"{f}_nominal"] = f

df_imp = pd.DataFrame(
    index=training_features + ["total"],
    columns=["significance", "dnn_shuffle", "mean_grad"],
)
df_imp[["significance", "dnn_shuffle", "mean_grad"]] = 0

counting_exp = df.loc[df.dataset.isin(signals), "wgt_nominal"].sum() / np.sqrt(
    df.loc[df.dataset.isin(signals), "wgt_nominal"].sum()
    + df.loc[df.dataset.isin(backgrounds), "wgt_nominal"].sum()
)
print("Counting exp.: ", counting_exp)

folds = [0, 1, 2, 3]
for i in folds:
    scalers_path = (
        f"{parameters['models_path']}/{channel}/scalers/scalers_{model_name}_{i}.npy"
    )
    scalers = np.load(scalers_path)

    folds_def = {"train": [0, 1], "val": [2], "eval": [3]}
    folds_shifted = {}
    for fname, folds in folds_def.items():
        folds_shifted[fname] = [(i + f) % 4 for f in folds]
    train_filter = df.event.mod(4).isin(folds_shifted["train"])
    val_filter = df.event.mod(4).isin(folds_shifted["val"])
    eval_filter = df.event.mod(4).isin(folds_shifted["eval"])

    dnn_model = Net(len(training_features))
    model_path = f"{parameters['models_path']}/{channel}/models/{model_name}_{i}.pt"
    dnn_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    dnn_model.eval()

    criterion = nn.BCELoss()

    mean_grads = get_mean_grads(
        df[eval_filter], features, scalers, dnn_model, criterion
    )
    # print(mean_grads)

    for feature in ["total"] + features:
        # for feature in ["total", "jj_mass_nominal", "jj_mass_log_nominal"]:
        ret = get_importance(
            df[eval_filter], features, feature, scalers, dnn_model, criterion
        )
        df_imp.loc[feat_map[feature], "significance"] += ret[0]
        df_imp.loc[feat_map[feature], "dnn_shuffle"] += ret[1]

    # df_imp.significance = df_imp.significance / counting_exp
    df_imp.loc[:, "mean_grad"] += mean_grads

df_imp["significance"] = np.sqrt(df_imp["significance"])
df_imp["dnn_shuffle"] = np.sqrt(df_imp["dnn_shuffle"])
df_imp["mean_grad"] = df_imp["mean_grad"] / 4
df_imp.loc["total", "significance"] = df_imp.loc["total", "dnn_shuffle"]
df_imp.loc["total", "dnn_shuffle"] = counting_exp
df_imp = df_imp.sort_values("significance", ascending=False)

print(df_imp)
df_imp.to_pickle("plots/feat_imp.pkl")
