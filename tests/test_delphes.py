import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time

from coffea.processor import dask_executor, run_uproot_job
from coffea.nanoevents import DelphesSchema
from delphes.preprocessor import get_fileset
from delphes.processor import DimuonProcessorDelphes

import dask
from dask.distributed import Client

__all__ = ["Client"]


def almost_equal(a, b):
    return abs(a - b) < 10e-6


if __name__ == "__main__":
    tick = time.time()

    parameters = {
        "lumi": 3000000,
    }
    parameters["client"] = dask.distributed.Client(
        processes=True,
        n_workers=1,
        threads_per_worker=1,
        memory_limit="2.9GB",
    )
    print("Client created")

    file_name = "ggh_delphes.root"
    file_path = f"{os.getcwd()}/tests/samples/{file_name}"
    datasets = {"ggh_powheg": file_path}

    fileset = get_fileset(datasets, parameters)

    executor = dask_executor
    executor_args = {
        "client": parameters["client"],
        "schema": DelphesSchema,
        "use_dataframes": True,
        "retries": 0,
    }
    output = run_uproot_job(
        fileset,
        "Delphes",
        DimuonProcessorDelphes(),
        executor,
        executor_args=executor_args,
        chunksize=10000,
    )
    df = output.compute()
    print(df)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")

    dimuon_mass = df.loc[df.event == 20002, "dimuon_mass"].values[0]
    jj_mass = df.loc[df.event == 20011, "jj_mass"].values[0]
    assert df.shape == (59, 63)
    assert almost_equal(dimuon_mass, 125.239198688)
    assert almost_equal(jj_mass, 78.593476)
