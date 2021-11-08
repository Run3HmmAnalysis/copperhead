from dask.distributed import Client
import dask.dataframe as dd
import pandas as pd
import glob
import time

import sys

[sys.path.append(i) for i in [".", ".."]]

from python.variable import Variable
from python.convert import to_histograms
from python.plotter import plotter
from python.io import mkdir

from trainer import Trainer
from dnn_models import test_model_1, test_model_2

training_datasets = {
    "background": ["dy_m100_mg", "ttbar_dl"],
    "signal": ["ggh_powheg", "vbf_powheg"],
    "ignore": [
        "tttj",
        "tttt",
        "tttw",
        "ttwj",
        "ttww",
        "ttz",
        "st_s",
        "st_t_antitop",
        "st_tw_top",
        "st_tw_antitop",
        "zz_2l2q",
    ],
}
# something similar to Run 2 VBF DNN
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

# exactly like in Run 2 ggH BDT
features = [
    "dimuon_pt",
    "dimuon_rap",
    "dimuon_cos_theta_cs",
    "dimuon_phi_cs",
    "mu1_pt_over_mass",
    "mu1_eta",
    "mu2_pt_over_mass",
    "mu2_eta",
    "jet1_pt",
    "jet1_eta",
    "mmj1_dEta",
    "mmj1_dPhi",
    "jet2_pt",
    "jet2_eta",
    "mmj2_dEta",
    "mmj2_dPhi",
    "jj_dEta",
    "jj_dPhi",
    "jj_mass",
    "zeppenfeld",
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

    # We will train separate methods depending on njets
    # (in VBF category it will always be 2 or more)
    categories = {
        "cat_0jets": df.njets == 0,
        "cat_1jet": df.njets == 1,
        "cat_2orMoreJets": df.njets >= 2,
    }
    trainers = {}
    for cat_name, cat_filter in categories.items():
        t_start = time.time()
        out_dir = "/home/dkondra/hmumu-coffea-dev/mva/hmumu-coffea/dev/mva_output/"
        mkdir(out_dir)
        out_dir = f"{out_dir}/{cat_name}"
        mkdir(out_dir)
        parameters["plots_path"] = out_dir

        trainers[cat_name] = Trainer(
            df=df[cat_filter],
            cat_name=cat_name,
            ds_dict=training_datasets,
            features=features,
            out_path=out_dir,
        )
        trainers[cat_name].add_models({"test1": test_model_1, "test2": test_model_2})
        trainers[cat_name].run_training(client)
        trainers[cat_name].run_evaluation(client)

        trainers[cat_name].plot_roc_curves()
        trainers[cat_name].df["channel"] = "vbf"  # temporary

        hist_df = to_histograms(client, parameters, trainers[cat_name].df)
        plotter(client, parameters, hist_df)
        elapsed = round(time.time() - t_start, 3)
        print(f"Category {cat_name}: done in {elapsed} s.")


if __name__ == "__main__":
    parameters = {
        "ncpus": 20,
        "variables_lookup": {
            "test1_score": Variable("test1_score", "test1_score", 50, 0, 1),
            "test2_score": Variable("test2_score", "test2_score", 50, 0, 1),
        },
        "years": ["snowmass"],
        "regions": ["h-peak"],
        "channels": ["vbf"],
        "has_variations": False,
        "save_hists": False,
        "hist_vars": ["test1_score", "test2_score"],
        "plot_vars": ["test1_score", "test2_score"],
        "plot_ratio": False,
        # "datasets": ["dy_m100_mg", "ttbar_dl", "ggh_powheg", "vbf_powheg"],
        "grouping": {
            "ggh_powheg": "ggH",
            "vbf_powheg": "VBF",
            "dy_m100_mg": "DY",
            "ttbar_dl": "TTbar",
            "tttj": "TTbar",
            "tttt": "TTbar",
            "tttw": "TTbar",
            "ttwj": "TTbar",
            "ttww": "TTbar",
            "ttz": "TTbar",
            "st_s": "Single top",
            # "st_t_top": "Single top",
            "st_t_antitop": "Single top",
            "st_tw_top": "Single top",
            "st_tw_antitop": "Single top",
            "zz_2l2q": "VV",
        },
        "plot_groups": {
            "stack": ["DY", "TTbar", "Single Top", "VV"],
            "step": ["VBF", "ggH"],
        },
        "14TeV_label": True,
        "save_plots": True,
    }
    paths = []
    all_ds = []
    for group, ds_list in training_datasets.items():
        for dataset in ds_list:
            all_ds.append(dataset)
            paths.extend(
                glob.glob(f"/depot/cms/hmm/coffea/snowmass_nov3/{dataset}/*.parquet")
            )
    parameters["datasets"] = all_ds

    client = Client(
        processes=True,
        n_workers=parameters["ncpus"],
        threads_per_worker=1,
        memory_limit="4GB",
    )

    tick = time.time()
    workflow(client, paths, parameters)
    elapsed = round(time.time() - tick, 3)
    print(f"Completed in {elapsed} s.")
