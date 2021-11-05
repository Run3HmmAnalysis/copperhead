from dask.distributed import Client
import dask.dataframe as dd
import pandas as pd
import glob

from train_dnn import train_dnn
from train_bdt import train_bdt

training_datasets = {
    "background": ["dy_m100_mg", "ttbar_dl"],
    "signal": ["ggh_powheg", "vbf_powheg"],
}
# something similar to Run 2
features = [
    "dimuon_mass",
    "dimuon_pt",
    "dimuon_pt_log",
    "dimuon_eta",
    "dimuon_cos_theta_cs",
    "dimuon_phi_cs",
    "jet1_pt",
    "jet1_eta",
    "jet1_phi",
    "jet2_pt",
    "jet2_eta",
    "jet2_phi",
    "jj_mass",
    "jj_mass_log",
    "jj_dEta",
    "rpt",
    "ll_zstar_log",
    "mmj_min_dEta",
]


def load_data(path):
    if len(path) > 0:
        df = dd.read_parquet(path)
    else:
        df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    return df


def workflow(client, paths, parameters):
    # Load dataframes
    df_future = client.map(load_data, paths)
    df_future = client.gather(df_future)

    df = dd.concat([d for d in df_future if len(d.columns) > 0])
    df = df.compute()
    df.reset_index(inplace=True, drop=True)

    # Convert dictionary of datasets to a more useful dataframe
    df_info = prepare_info(training_datasets)
    df = df[df.dataset.isin(df_info.dataset.unique())]

    # Assign numerical classes to each event
    cls_map = dict(df_info[["dataset", "iclass"]].values)
    df["class"] = df.dataset.map(cls_map)

    # We will train separate methods depending on njets
    # (in VBF category it will always be 2 or more)
    njets_cats = {
        "cat_0jets": df.njets == 0,
        "cat_1jet": df.njets == 1,
        "cat_2orMoreJets": df.njets >= 2,
    }
    for cat_name, cat_filter in njets_cats.items():
        if parameters["train_dnn"]:
            train_dnn(df[cat_filter], cat_name, df_info, features)
        if parameters["train_bdt"]:
            train_bdt(df[cat_filter], cat_name, df_info, features)


def prepare_info(ds_dict):
    df_info = pd.DataFrame()
    for icls, (cls, ds_list) in enumerate(ds_dict.items()):
        for ds in ds_list:
            df_info.loc[ds, "dataset"] = ds
            df_info.loc[ds, "class"] = cls
            df_info.loc[ds, "iclass"] = icls
    df_info["iclass"] = df_info["iclass"].astype(int)
    return df_info


if __name__ == "__main__":
    parameters = {"ncpus": 20, "train_dnn": True, "train_bdt": False}
    paths = []
    for group, ds_list in training_datasets.items():
        for dataset in ds_list:
            paths.extend(
                glob.glob(f"/depot/cms/hmm/coffea/snowmass_nov3/{dataset}/0*.parquet")
            )

    client = Client(
        processes=True,
        n_workers=parameters["ncpus"],
        threads_per_worker=1,
        memory_limit="4GB",
    )

    workflow(client, paths, parameters)
