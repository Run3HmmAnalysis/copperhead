import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time

from coffea.processor import DaskExecutor, Runner
from coffea.nanoevents import NanoAODSchema

from python.io import load_dataframe
from config.variables import variables_lookup

from stage1.preprocessor import SamplesInfo
from stage1.processor import DimuonProcessor
from stage2.postprocessor import process_partitions
from stage3.plotter import plotter
from test_tools import almost_equal

import dask
from dask.distributed import Client

__all__ = ["Client"]


parameters = {
    "ncpus": 1,
    "years": [2018],
    "datasets": ["vbf_powheg"],
    "channels": ["vbf"],
    "regions": ["h-peak"],
    "hist_vars": ["dimuon_mass"],
    "plot_vars": ["dimuon_mass"],
    "return_hist": True,
    "plot_ratio": True,
    "variables_lookup": variables_lookup,
    "grouping": {"vbf_powheg": "VBF"},
    "plot_groups": {"stack": [], "step": ["VBF"], "errorbar": []},
}

if __name__ == "__main__":
    tick = time.time()

    client = dask.distributed.Client(
        processes=True, n_workers=1, threads_per_worker=1, memory_limit="2.9GB"
    )
    print("Client created")

    file_name = "vbf_powheg_dipole_NANOV10_2018.root"
    file_path = f"{os.getcwd()}/tests/samples/{file_name}"
    dataset = {"vbf_powheg": file_path}

    # Stage 1
    samp_info = SamplesInfo(xrootd=False)
    samp_info.paths = dataset
    samp_info.year = "2018"
    samp_info.load("vbf_powheg", use_dask=False)
    samp_info.lumi_weights["vbf_powheg"] = 1.0
    print(samp_info.fileset)

    executor_args = {"client": client, "use_dataframes": True, "retries": 0}
    processor_args = {"samp_info": samp_info, "do_timer": False, "do_btag_syst": False}

    executor = DaskExecutor(**executor_args)
    run = Runner(executor=executor, schema=NanoAODSchema, chunksize=10000)
    out_df = run(
        samp_info.fileset,
        "Events",
        processor_instance=DimuonProcessor(**processor_args),
    )

    out_df = out_df.compute()

    dimuon_mass = out_df.loc[out_df.event == 2, "dimuon_mass"].values[0]
    jj_mass = out_df.loc[out_df.event == 2, "jj_mass nominal"].values[0]
    print(out_df.shape)
    assert out_df.shape == (21806, 116)
    assert almost_equal(dimuon_mass, 124.16069531)
    assert almost_equal(jj_mass, 1478.3898375)

    # Stage 2
    df = load_dataframe(client, parameters, inputs=out_df)
    out_hist = process_partitions(client, parameters, df=df)

    assert almost_equal(
        out_hist.loc[out_hist.variation == "nominal", "yield"].values[0],
        31778.21631,
        precision=0.01,
    )

    # Stage 3
    out_plot = plotter(client, parameters, out_hist)

    assert almost_equal(out_plot[0], 31778.215944)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")
