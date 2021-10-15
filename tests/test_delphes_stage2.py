import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time

import dask
from dask.distributed import Client

from delphes.postprocessor import workflow, plotter, grouping_alt
from python.utils import almost_equal

__all__ = ["dask"]


parameters = {
    "ncpus": 1,
    "label": "test",
    "path": "/depot/cms/hmm/coffea/",
    "hist_path": "/depot/cms/hmm/coffea/snowmass_histograms/",
    "plots_path": "./plots_test/snowmass/",
    "years": ["snowmass"],
    "channels": ["vbf", "vbf_01j", "vbf_2j"],
    "regions": ["h-peak", "h-sidebands"],
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

    out = workflow(client, [path], parameters)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")
    assert almost_equal(out["hist"][0]["h-peak", "vbf", "value", :].sum(), 4515.761427143451)
