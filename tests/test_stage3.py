import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time
import pickle
import pandas as pd

import dask
from dask.distributed import Client

from python.io import load_dataframe
from config.variables import variables_lookup
from stage3.plotter import plotter
from test_tools import almost_equal

__all__ = ["dask"]


parameters = {
    "ncpus": 1,
    "years": [2016],
    "datasets": ["vbf_powheg_dipole"],
    "channels": ["vbf"],
    "regions": ["h-peak"],
    "plot_vars": ["dimuon_mass"],
    "plot_ratio": True,
    "variables_lookup": variables_lookup,
    "grouping": {"vbf_powheg_dipole": "VBF"},
    "plot_groups": {"stack": [], "step": ["VBF"], "errorbar": []},
}


if __name__ == "__main__":
    tick = time.time()

    client = Client(
        processes=True, n_workers=1, threads_per_worker=1, memory_limit="4GB"
    )

    file_name = "vbf_dimuon_mass_hist.pickle"
    path = f"{os.getcwd()}/tests/samples/{file_name}"

    with open(path, "rb") as handle:
        hist = pickle.load(handle)
    hist_df = pd.DataFrame(
        [
            {
                "year": 2016,
                "var_name": "dimuon_mass",
                "dataset": "vbf_powheg_dipole",
                "hist": hist,
            }
        ]
    )
    out_plot = plotter(client, parameters, hist_df)

    path_unbinned = f"{os.getcwd()}/tests/samples/vbf_stage2_unbinned.parquet"
    df = load_dataframe(client, parameters, inputs=[path_unbinned])
    print(df.compute())

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")

    assert almost_equal(
        out_plot[0],
        0.7066942,
    )
