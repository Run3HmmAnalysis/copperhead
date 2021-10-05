import sys
import time
import argparse
import traceback
from functools import partial

import coffea
from coffea.nanoevents import DelphesSchema
from coffea.processor import dask_executor, run_uproot_job

from python.utils import mkdir
from delphes.preprocessor import get_fileset
from delphes.processor import DimuonProcessorDelphes
from delphes.datasets import datasets

from dask.distributed import Client
# dask.config.set({"temporary-directory": "/depot/cms/hmm/dask-temp/"})

parser = argparse.ArgumentParser()
# Slurm cluster IP to use. If not specified, will create a local cluster
parser.add_argument("-sl", "--slurm", dest="slurm_port",
                    default=None, action='store',
                    help='Slurm cluster port (if not specified, '
                    'will create a local cluster)')
parser.add_argument("-l", "--label", dest="label", default="test",
                    action='store',
                    help='Unique run label (to create output path)')
parser.add_argument("-ch", "--chunksize", dest="chunksize",
                    default=100000, action='store',
                    help='Approximate chunk size')
parser.add_argument("-mch", "--maxchunks", dest="maxchunks", default=-1,
                    action='store',
                    help='Max. number of chunks')

args = parser.parse_args()

node_ip = '128.211.149.133'  # hammer-c000
# node_ip = '128.211.149.140' # hammer-c007
dash_local = f'{node_ip}:34875'

if args.slurm_port is None:
    local_cluster = True
    slurm_cluster_ip = ''
else:
    local_cluster = False
    slurm_cluster_ip = f'{node_ip}:{args.slurm_port}'

mch = None if int(args.maxchunks) < 0 else int(args.maxchunks)

year = 'snowmass'

parameters = {
    'year': year,
    'label': args.label,
    'global_out_path': '/depot/cms/hmm/coffea/',
    'out_path': f'{year}_{args.label}',
    'server': '/mnt/hadoop/',
    'chunksize': int(args.chunksize),
    'maxchunks': mch,
    'local_cluster': local_cluster,
    'slurm_cluster_ip': slurm_cluster_ip,
    'client': None,
    'lumi': 3000000,
}

parameters['out_dir'] = f"{parameters['global_out_path']}/{parameters['out_path']}"


def saving_func(output, out_dir):
    from dask.distributed import get_worker
    name = None
    for key, task in get_worker().tasks.items():
        if task.state == "executing":
            name = key[-32:]
    if not name:
        return
    for ds in output.s.unique():
        df = output[output.s == ds]
        if df.shape[0] == 0:
            return
        mkdir(f'{out_dir}/{ds}')
        df.to_parquet(
            path=f'{out_dir}/{ds}/{name}.parquet',
        )


def submit_job(arg_set, parameters):
    mkdir(parameters['out_dir'])
    out_dir = f"{parameters['out_dir']}/"
    mkdir(out_dir)

    executor = dask_executor
    executor_args = {
        'client': parameters['client'],
        'schema': DelphesSchema,
        'retries': 0
    }
    processor_args = {
        'apply_to_output': partial(saving_func, out_dir=out_dir),
    }

    try:
        run_uproot_job(
            parameters['fileset'], 'Delphes',
            DimuonProcessorDelphes(**processor_args),
            executor, executor_args=executor_args,
            chunksize=parameters['chunksize'],
            maxchunks=parameters['maxchunks']
        )
    except Exception as e:
        tb = traceback.format_exc()
        return 'Failed: ' + str(e) + ' ' + tb

    return 'Success!'


if __name__ == "__main__":
    tick = time.time()
    if parameters['local_cluster']:
        parameters['client'] = Client(
            processes=True,
            n_workers=40,
            dashboard_address=dash_local,
            threads_per_worker=1,
            memory_limit='2.9GB',
        )
    else:
        parameters['client'] = Client(
            parameters['slurm_cluster_ip'],
        )
    print('Client created')

    parameters['fileset'] = get_fileset(datasets, parameters)

    out = submit_job({}, parameters)
    print(out)

    elapsed = round(time.time() - tick, 3)
    print(f'Finished everything in {elapsed} s.')
