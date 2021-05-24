import os
import sys
import time

import coffea.processor as processor
from coffea.processor import dask_executor, run_uproot_job
from python.dimuon_processor_pandas import DimuonProcessor
from python.samples_info import SamplesInfo

import dask
from dask.distributed import Client

__all__ = ['Client']

[sys.path.append(i) for i in ['.', '..']]


def almost_equal(a, b):
    return (abs(a-b) < 10e-6)


if __name__ == "__main__":
    tick = time.time()

    client = dask.distributed.Client(
        processes=True,
        n_workers=1,
        threads_per_worker=1,
        memory_limit='2.9GB',
    )
    print('Client created')

    file_name = "vbf_powheg_dipole_NANOV10_2018.root"
    file_path = f"{os.getcwd()}/tests/samples/{file_name}"
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

    dimuon_mass = df.loc[df.event == 2, 'dimuon_mass'].values[0]
    jj_mass = df.loc[df.event == 2, 'jj_mass nominal'].values[0]
    assert(df.shape == (8594, 100))
    assert(almost_equal(dimuon_mass, 124.16069531))
    assert(almost_equal(jj_mass, 1478.3898375))
