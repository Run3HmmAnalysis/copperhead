from dask.distributed import Client
import dask.dataframe as dd
import pandas as pd
import glob

from train_dnn import train_dnn
from train_bdt import train_bdt

training_datasets = {
    "signal": ["ggh_powheg", "vbf_powheg"],
    "background": ["dy_m100_mg", "ttjets_dl"],
}

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

    print(df)

    if parameters["train_dnn"]:
        train_dnn(df, training_datasets, training_features)

    if parameters["train_bdt"]:
        train_bdt(df, training_datasets, training_features)


if __name__ == "__main__":
    parameters = {"ncpus": 1, "train_dnn": True, "train_bdt": False}
    paths = glob.glob("/depot/cms/hmm/coffea/snowmass_oct14/dy_m100_mg/00*.parquet")

    client = Client(
        processes=True,
        n_workers=parameters["ncpus"],
        threads_per_worker=1,
        memory_limit="4GB",
    )

    workflow(client, paths, parameters)
