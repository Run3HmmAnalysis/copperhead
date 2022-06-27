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
    "datasets": ["ewk_lljj_mll105_160_ptj0"],
    "channels": ["vbf"],
    "regions": ["h-peak"],
    "hist_vars": ["dimuon_mass"],
    "plot_vars": ["dimuon_mass"],
    "return_hist": True,
    "plot_ratio": True,
    "variables_lookup": variables_lookup,
    "grouping": {"ewk_lljj_mll105_160_ptj0": "EWK"},
    "plot_groups": {"stack": ["EWK"], "step": [], "errorbar": []},
}

if __name__ == "__main__":
    tick = time.time()

    client = dask.distributed.Client(
        processes=True, n_workers=1, threads_per_worker=1, memory_limit="2.9GB"
    )
    print("Client created")

    file_name = "ewk_lljj_mll105_160_ptj0_NANOV10_2018.root"
    file_path = f"{os.getcwd()}/tests/samples/{file_name}"
    dataset = {"ewk_lljj_mll105_160_ptj0": file_path}

    # Stage 1
    samp_info = SamplesInfo(xrootd=False)
    samp_info.paths = dataset
    samp_info.year = "2018"
    samp_info.load("ewk_lljj_mll105_160_ptj0", use_dask=False)
    samp_info.lumi_weights["ewk_lljj_mll105_160_ptj0"] = 1.0
    print(samp_info.fileset)

    executor_args = {"client": client, "use_dataframes": True, "retries": 0}
    processor_args = {
        "samp_info": samp_info,
        "do_btag_syst": False,
        "regions": ["h-peak"],
    }

    executor = DaskExecutor(**executor_args)
    run = Runner(executor=executor, schema=NanoAODSchema, chunksize=10000)
    out_df = run(
        samp_info.fileset,
        "Events",
        processor_instance=DimuonProcessor(**processor_args),
    )

    out_df = out_df.compute()
    print(out_df)
    dimuon_mass = out_df.loc[out_df.event == 2254006, "dimuon_mass"].values[0]
    jj_mass = out_df.loc[out_df.event == 2254006, "jj_mass_nominal"].values[0]

    assert out_df.shape == (391, 122)
    assert almost_equal(dimuon_mass, 117.1209375)
    assert almost_equal(jj_mass, 194.5646039)

    # Stage 2
    df = load_dataframe(client, parameters, inputs=out_df)
    out_hist = process_partitions(client, parameters, df=df)

    print(out_hist.loc[out_hist.variation == "nominal", "yield"].values[0])
    assert almost_equal(
        out_hist.loc[out_hist.variation == "nominal", "yield"].values[0],
        46.7871466,
        precision=0.01,
    )

    # Stage 3
    out_plot = plotter(client, parameters, out_hist)
    print(out_plot[0])
    assert almost_equal(out_plot[0], 701.8071994)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")
