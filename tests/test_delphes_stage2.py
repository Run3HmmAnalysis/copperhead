import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time

import dask
from dask.distributed import Client

from delphes.postprocessor import workflow, plotter
from python.utils import almost_equal

__all__ = ["dask"]


parameters = {
    "ncpus": 1,
    "years": ["snowmass"],
    "channels": ["vbf"],
    "regions": ["h-peak"],
    "save_hists": False,
    "save_plots": False,
}


if __name__ == "__main__":
    tick = time.time()

    client = Client(
        processes=True,
        n_workers=1,
        threads_per_worker=1,
        memory_limit="4GB",
    )

    parameters["hist_vars"] = ["dimuon_mass"]
    parameters["plot_vars"] = ["dimuon_mass"]
    parameters["datasets"] = ["dy_m100_mg"]

    file_name = "dy_delphes_stage1_output.parquet"
    path = f"{os.getcwd()}/tests/samples/{file_name}"

    out_hist = workflow(client, [path], parameters)
    out_plot = plotter(client, parameters, hist_df=out_hist)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")
    assert almost_equal(
        out_hist["hist"][0]["h-peak", "vbf", "value", :].sum(), 4515.761427143451
    )
    assert almost_equal(sum(out_plot), 4515.761427143451)
