import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time

import dask
from dask.distributed import Client

from nanoaod.postprocessor import workflow
from python.utils import almost_equal
from plotting.plotter import plotter

__all__ = ["dask"]


parameters = {
    "ncpus": 1,
    "years": [2018],
    "datasets": ["dy_m105_160_vbf_amc"],
    "channels": ["vbf"],
    "regions": ["h-peak"],
    "syst_variations": ["nominal"],
    "dnn_models": [],
    "bdt_models": [],
    "hist_vars": ["dimuon_mass"],
    "plot_vars": ["dimuon_mass"],
    "save_hists": False,
    "save_plots": False,
    "plot_ratio": True,
    "14TeV_label": False,
    "grouping": {
        "dy_m105_160_vbf_amc": "DY",
    },
    "stack_groups": ["DY"],
    "data_group": [],
    "step_groups": [],
}


if __name__ == "__main__":
    tick = time.time()

    client = Client(
        processes=True,
        n_workers=1,
        threads_per_worker=1,
        memory_limit="4GB",
    )

    file_name = "dy_nanoaod_stage1_output.parquet"
    path = f"{os.getcwd()}/tests/samples/{file_name}"

    out_hist = workflow(client, [path], parameters)
    out_plot = plotter(client, parameters, hist_df=out_hist)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")

    assert almost_equal(
        out_hist["hist"][0]["h-peak", "vbf", "nominal", "value", :].sum(),
        6.761677545416264,
    )
    assert almost_equal(sum(out_plot), 6.761677545416264)
