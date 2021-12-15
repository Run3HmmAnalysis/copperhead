import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time

from coffea.processor import DaskExecutor, Runner
from coffea.nanoevents import DelphesSchema
from delphes.preprocessor import get_fileset
from delphes.processor import DimuonProcessorDelphes
from test_tools import almost_equal

import dask
from dask.distributed import Client

__all__ = ["Client"]


if __name__ == "__main__":
    tick = time.time()

    parameters = {"lumi": 3000000}
    client = dask.distributed.Client(
        processes=True, n_workers=1, threads_per_worker=1, memory_limit="2.9GB"
    )
    print("Client created")

    file_name = "ggh_delphes.root"
    file_path = f"{os.getcwd()}/tests/samples/{file_name}"
    datasets = {"ggh_powheg": file_path}

    fileset = get_fileset(client, datasets, parameters)

    executor_args = {"client": client, "use_dataframes": True, "retries": 0}
    executor = DaskExecutor(**executor_args)
    run = Runner(executor=executor, schema=DelphesSchema, chunksize=10000)
    output = run(fileset, "Delphes", processor_instance=DimuonProcessorDelphes())

    df = output.compute()
    print(df)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")

    dimuon_mass = df.loc[df.event == 20002, "dimuon_mass"].values[0]
    jj_mass = df.loc[df.event == 20011, "jj_mass"].values[0]
    assert df.shape == (86, 63)
    assert almost_equal(dimuon_mass, 124.3369651)
    assert almost_equal(jj_mass, 78.593476)
