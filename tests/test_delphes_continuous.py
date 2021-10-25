import os
import sys

[sys.path.append(i) for i in [".", ".."]]
import time

from coffea.processor import dask_executor, run_uproot_job
from coffea.nanoevents import DelphesSchema
from delphes.preprocessor import get_fileset
from delphes.processor import DimuonProcessorDelphes
from delphes.postprocessor import load_dataframe
from delphes.config.variables import variables_lookup
from python.convert import to_histograms, to_templates
from python.plotter import plotter
from python.utils import almost_equal

import dask
from dask.distributed import Client

__all__ = ["Client"]


if __name__ == "__main__":
    tick = time.time()

    parameters = {
        "lumi": 3000000,
        "ncpus": 1,
        "years": ["snowmass"],
        "datasets": ["ggh_powheg"],
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
        "grouping": {"ggh_powheg": "ggH"},
        "plot_groups": {"stack": [], "step": ["ggH"], "errorbar": []},
    }

    client = dask.distributed.Client(
        processes=True, n_workers=1, threads_per_worker=1, memory_limit="2.9GB"
    )
    print("Client created")

    file_name = "ggh_delphes.root"
    file_path = f"{os.getcwd()}/tests/samples/{file_name}"
    datasets = {"ggh_powheg": file_path}

    fileset = get_fileset(client, datasets, parameters)

    executor = dask_executor
    executor_args = {
        "client": client,
        "schema": DelphesSchema,
        "use_dataframes": True,
        "retries": 0,
    }
    out_df = run_uproot_job(
        fileset,
        "Delphes",
        DimuonProcessorDelphes(),
        executor,
        executor_args=executor_args,
        chunksize=10000,
    )

    df = load_dataframe(client, parameters, inputs=out_df)
    out_hist = to_histograms(client, parameters, df=df)
    out_plot = plotter(client, parameters, hist_df=out_hist)
    out_tmp = to_templates(client, parameters, hist_df=out_hist)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")

    out_df = out_df.compute()
    dimuon_mass = out_df.loc[out_df.event == 20002, "dimuon_mass"].values[0]
    jj_mass = out_df.loc[out_df.event == 20011, "jj_mass"].values[0]
    assert out_df.shape == (59, 63)
    assert almost_equal(dimuon_mass, 125.239198688)
    assert almost_equal(jj_mass, 78.593476)
    slicer = {
        "region": "h-peak",
        "channel": "vbf",
        "val_sumw2": "value",
        "dimuon_mass": slice(None),
    }
    assert almost_equal(out_hist["hist"][0][slicer].sum(), 14911.835814002365)
    assert almost_equal(sum(out_plot), 14911.835814002365)
    assert almost_equal(sum(out_tmp), 14911.835814002365)
