import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time

import dask
from dask.distributed import Client

from stage2.postprocessor import load_dataframe, process_partitions
from config.variables import variables_lookup
from test_tools import almost_equal

__all__ = ["dask"]


parameters = {
    "ncpus": 1,
    "years": [2016],
    "datasets": ["dy_m105_160_amc"],
    "channels": ["vbf"],
    "regions": ["h-peak"],
    "syst_variations": ["nominal"],
    "dnn_models": [],
    "bdt_models": [],
    "hist_vars": ["dimuon_mass"],
    "save_hists": False,
    "variables_lookup": variables_lookup,
}


if __name__ == "__main__":
    tick = time.time()

    client = Client(
        processes=True, n_workers=1, threads_per_worker=1, memory_limit="4GB"
    )

    file_name = "dy_nanoaod_stage1_output.parquet"
    path = f"{os.getcwd()}/tests/samples/{file_name}"

    df = load_dataframe(client, parameters, inputs=[path])
    out_hist = process_partitions(client, parameters, df)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")

    assert almost_equal(
        out_hist.loc[out_hist.variation == "nominal", "yield"].values[0],
        0.14842246076249055,
    )
