import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time

import dask
from dask.distributed import Client

from python.io import load_dataframe
from config.variables import variables_lookup
from stage2.postprocessor import process_partitions
from test_tools import almost_equal

__all__ = ["dask"]


parameters = {
    "ncpus": 1,
    "years": [2016],
    "datasets": ["data_B"],
    "channels": ["ggh_0jets"],
    "regions": ["h-peak"],
    "hist_vars": ["dimuon_mass"],
    "variables_lookup": variables_lookup,
}


if __name__ == "__main__":
    tick = time.time()

    client = Client(
        processes=True, n_workers=1, threads_per_worker=1, memory_limit="4GB"
    )

    file_name = "data_B_stage1_output.parquet"
    path = f"{os.getcwd()}/tests/samples/{file_name}"

    df = load_dataframe(client, parameters, inputs=[path])
    out_hist = process_partitions(client, parameters, df)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")
    print(out_hist)
    assert almost_equal(
        out_hist.loc[out_hist.variation == "nominal", "yield"].values[0],
        21.0,
    )
