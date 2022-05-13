import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time

from coffea.processor import DaskExecutor, Runner
from coffea.nanoevents import NanoAODSchema

from stage1.processor import DimuonProcessor
from stage1.preprocessor import SamplesInfo
from test_tools import almost_equal

import dask
from dask.distributed import Client

__all__ = ["Client"]


if __name__ == "__main__":
    tick = time.time()

    client = dask.distributed.Client(
        processes=True, n_workers=1, threads_per_worker=1, memory_limit="2.9GB"
    )
    print("Client created")

    file_name = "ewk_lljj_mll105_160_ptj0_NANOV10_2018.root"
    file_path = f"{os.getcwd()}/tests/samples/{file_name}"
    dataset = {"test": file_path}

    samp_info = SamplesInfo(xrootd=False)
    samp_info.paths = dataset
    samp_info.year = "2018"
    samp_info.load("test", use_dask=False)
    samp_info.lumi_weights["test"] = 1.0
    print(samp_info.fileset)

    executor_args = {"client": client, "use_dataframes": True, "retries": 0}
    processor_args = {
        "samp_info": samp_info,
        "do_btag_syst": False,
        "regions": ["h-peak"],
    }

    executor = DaskExecutor(**executor_args)
    run = Runner(executor=executor, schema=NanoAODSchema, chunksize=10000)
    output = run(
        samp_info.fileset,
        "Events",
        processor_instance=DimuonProcessor(**processor_args),
    )

    df = output.compute()
    print(df)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")

    dimuon_mass = df.loc[df.event == 2254006, "dimuon_mass"].values[0]
    jj_mass = df.loc[df.event == 2254006, "jj_mass_nominal"].values[0]

    assert df.shape == (391, 120)
    assert almost_equal(dimuon_mass, 117.1209375)
    assert almost_equal(jj_mass, 194.5646039)
