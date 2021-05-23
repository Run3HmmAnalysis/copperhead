import os, sys
[sys.path.append(i) for i in ['.', '..']]

import time
import traceback

import coffea.processor as processor
from coffea.processor import dask_executor, run_uproot_job
from python.dimuon_processor_pandas import DimuonProcessor
from python.samples_info import SamplesInfo

import dask
import dask.dataframe as dd

from dask.distributed import Client

if __name__ == "__main__":
    tick = time.time()

    client = dask.distributed.Client(
        processes=True,
        n_workers=1,
        threads_per_worker=1,
        memory_limit='2.9GB',
    )
    print('Client created')

    file_path = f"{os.getcwd()}/tests/samples/vbf_powheg_dipole_NANOV10_2018.root"
    dataset = {'test': file_path}

    samp_info = SamplesInfo(xrootd=False)
    samp_info.paths = dataset
    samp_info.year = '2018'
    samp_info.load('test', use_dask=False)
    samp_info.lumi_weights['test'] = 1.
    executor = dask_executor
    executor_args = {
        'client': client,
        'schema': processor.NanoAODSchema,
        'use_dataframes': True,
        'retries': 0
    }
    processor_args = {
        'samp_info': samp_info,
        'do_timer': False,
        'do_btag_syst': False,
    }  
    print(samp_info.fileset)
    output = run_uproot_job(samp_info.fileset, 'Events',
                            DimuonProcessor(**processor_args),
                            executor, executor_args=executor_args,
                            chunksize=10000)
    df = output.compute()
    print(df)

    elapsed = round(time.time() - tick, 3)
    print(f'Finished everything in {elapsed} s.')
