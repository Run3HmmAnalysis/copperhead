import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time

import dask
from dask.distributed import Client

from nanoaod.postprocessor import load_dataframe
from nanoaod.config.variables import variables_lookup
from python.convert import to_histograms
from python.plotter import plotter
from tests.test_tools import almost_equal

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
    "has_variations": True,
    "variables_lookup": variables_lookup,
    "grouping": {"dy_m105_160_vbf_amc": "DY"},
    "plot_groups": {"stack": "DY", "step": [], "errorbar": []},
}


if __name__ == "__main__":
    tick = time.time()

    client = Client(
        processes=True, n_workers=1, threads_per_worker=1, memory_limit="4GB"
    )

    file_name = "dy_nanoaod_stage1_output.parquet"
    path = f"{os.getcwd()}/tests/samples/{file_name}"

    df = load_dataframe(client, parameters, inputs=[path])
    out_hist = to_histograms(client, parameters, df=df)
    out_plot = plotter(client, parameters, hist_df=out_hist)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")

    slicer = {
        "region": "h-peak",
        "channel": "vbf",
        "variation": "nominal",
        "val_sumw2": "value",
        "dimuon_mass": slice(None),
    }
    assert almost_equal(out_hist["hist"][0][slicer].sum(), 6.761677545416264)
    assert almost_equal(sum(out_plot), 6.761677545416264)
