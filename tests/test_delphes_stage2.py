import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time

import dask
from dask.distributed import Client

from delphes.postprocessor import load_dataframe
from delphes.config.variables import variables_lookup
from python.convert import to_histograms, to_templates
from python.utils import almost_equal
from plotting.plotter import plotter

__all__ = ["dask"]


parameters = {
    "ncpus": 1,
    "years": ["snowmass"],
    "datasets": ["dy_m100_mg"],
    "channels": ["vbf"],
    "regions": ["h-peak"],
    "hist_vars": ["dimuon_mass"],
    "plot_vars": ["dimuon_mass"],
    "save_hists": False,
    "save_plots": False,
    "save_templates": False,
    "plot_ratio": False,
    "14TeV_label": True,
    "has_variations": False,
    "variables_lookup": variables_lookup,
    "grouping": {
        "dy_m100_mg": "DY",
    },
    "plot_groups": {
        "stack": "DY",
        "step": [],
        "data": [],
    },
}


if __name__ == "__main__":
    tick = time.time()

    client = Client(
        processes=True,
        n_workers=1,
        threads_per_worker=1,
        memory_limit="4GB",
    )

    file_name = "dy_delphes_stage1_output.parquet"
    path = f"{os.getcwd()}/tests/samples/{file_name}"

    out_df = load_dataframe(client, parameters, inputs=[path])
    out_hist = to_histograms(client, parameters, df=out_df)
    out_plot = plotter(client, parameters, hist_df=out_hist)
    out_tmp = to_templates(client, parameters, hist_df=out_hist)
    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")

    slicer = {
        "region": "h-peak",
        "channel": "vbf",
        "val_sumw2": "value",
        "dimuon_mass": slice(None),
    }
    assert almost_equal(out_hist["hist"][0][slicer].sum(), 4515.761427143451)
    assert almost_equal(sum(out_plot), 4515.761427143451)
    assert almost_equal(sum(out_tmp), 4515.761427143451)
