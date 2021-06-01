import os
import time
import argparse
import traceback

import coffea.processor as processor
from python.processor import DimuonProcessor
from python.preprocessor import load_samples
from python.utils import mkdir
from config.parameters import parameters as pars

import dask
from dask.distributed import Client
dask.config.set({"temporary-directory": "/depot/cms/hmm/dask-temp/"})

import pyspark.sql
from pyarrow.util import guid
from coffea.processor.spark.detail import (
    _spark_initialize,
    _spark_stop,
)
from pyspark import TaskContext
from functools import partial

os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"

from coffea.processor import run_spark_job
from coffea.processor.spark.spark_executor import spark_executor

parser = argparse.ArgumentParser()
# Slurm cluster IP to use. If not specified, will create a local cluster
parser.add_argument("-sl", "--slurm", dest="slurm_port",
                    default=None, action='store',
                    help='Slurm cluster port (if not specified, '
                    'will create a local cluster)')
parser.add_argument("-y", "--year", dest="year", default='2016',
                    action='store',
                    help='Year to process (2016, 2017 or 2018)')
parser.add_argument("-l", "--label", dest="label", default="test_march",
                    action='store',
                    help='Unique run label (to create output path)')
parser.add_argument("-ch", "--chunksize", dest="chunksize",
                    default=100000, action='store',
                    help='Approximate chunk size')
parser.add_argument("-mch", "--maxchunks", dest="maxchunks", default=-1,
                    action='store',
                    help='Max. number of chunks')
parser.add_argument("-jec", "--jec", dest="jec_unc", default=False,
                    action='store_true',
                    help='Enable JEC/JER uncertainties')

args = parser.parse_args()

node_ip = '128.211.149.133'
dash_local = f'{node_ip}:34875'


if args.slurm_port is None:
    local_cluster = True
    slurm_cluster_ip = ''
else:
    local_cluster = False
    slurm_cluster_ip = f'{node_ip}:{args.slurm_port}'

mch = None if int(args.maxchunks) < 0 else int(args.maxchunks)
if args.jec_unc:
    pt_variations = (
        ['nominal'] +
        pars['jec_variations'][args.year] +
        pars['jer_variations'][args.year]
    )
else:
    pt_variations = ['nominal']

parameters = {
    'year': args.year,
    'label': args.label,
    'global_out_path': '/depot/cms/hmm/coffea/',
    'out_path': f'{args.year}_{args.label}',
    'server': 'root://xrootd.rcac.purdue.edu/',
    'datasets_from': 'purdue',
    'pt_variations': pt_variations,
    'chunksize': int(args.chunksize),
    'maxchunks': mch,
    'save_output': True,
    'local_cluster': local_cluster,
    'slurm_cluster_ip': slurm_cluster_ip,
    'client': None,
}

parameters['out_dir'] = f"{parameters['global_out_path']}/"\
                        f"{parameters['out_path']}"


def saving_func(output, out_dir):
    ctx = TaskContext()
    name = f'part_{ctx.partitionId()}'

    for ds in output.s.unique():
        df = output[output.s == ds]
        if df.shape[0] == 0:
            return
        mkdir(f'{out_dir}/{ds}')
        path = f'{out_dir}/{ds}/{name}.parquet'
        df.to_parquet(
            path=path,
        )
        print(f'Saved to {path}')


def submit_job(arg_set, parameters):
    mkdir(parameters['out_dir'])
    if parameters['pt_variations'] == ['nominal']:
        out_dir = f"{parameters['out_dir']}/"
    else:
        out_dir = f"{parameters['out_dir']}_jec/"
    mkdir(out_dir)

    executor = spark_executor
    executor_args = {
        'schema': processor.NanoAODSchema,
        # 'use_dataframes': True,
        "file_type": "root"
    }
    processor_args = {
        'samp_info': parameters['samp_infos'],
        'do_timer': False,
        'do_btag_syst': False,
        'pt_variations': parameters['pt_variations'],
        'apply_to_output': partial(saving_func, out_dir=out_dir),
    }
    try:
        run_spark_job(
            parameters['samp_infos'].fileset,
            DimuonProcessor(**processor_args),
            executor, spark=parameters['spark'],
            thread_workers=32, partitionsize=parameters['chunksize'],
            executor_args=executor_args,
        )
        _spark_stop(parameters['spark'])
    except Exception as e:
        tb = traceback.format_exc()
        return 'Failed: ' + str(e) + ' ' + tb

    return 'Success!'


if __name__ == "__main__":
    tick = time.time()
    smp = {
        # 'single_file': [
        #     'test_file',
        # ],
        'data': [
            'data_A',
            'data_B',
            'data_C',
            'data_D',
            'data_E',
            'data_F',
            'data_G',
            'data_H',
        ],
        'signal': [
            'ggh_amcPS',
            'vbf_powhegPS',
            'vbf_powheg_herwig',
            'vbf_powheg_dipole'
        ],
        'main_mc': [
            'dy_m105_160_amc',
            'dy_m105_160_vbf_amc',
            'ewk_lljj_mll105_160_py',
            'ewk_lljj_mll105_160_ptj0',
            'ewk_lljj_mll105_160_py_dipole',
            'ttjets_dl',
        ],
        'other_mc': [
            'ttjets_sl', 'ttz', 'ttw',
            'st_tw_top', 'st_tw_antitop',
            'ww_2l2nu', 'wz_2l2q',
            'wz_3lnu',
            'wz_1l1nu2q',
            'zz',
        ],
    }

    if parameters['local_cluster']:
        parameters['client'] = dask.distributed.Client(
            processes=True,
            # n_workers=min(mch, 23),
            n_workers=48,
            dashboard_address=dash_local,
            threads_per_worker=1,
            memory_limit='2.9GB',
        )
    else:
        parameters['client'] = Client(
            parameters['slurm_cluster_ip'],
        )
    print('Client created')

    spark_config = (
        pyspark.sql.SparkSession.builder.appName(
            "spark-executor-test-%s" % guid()
        )
        .master("local[*]")
        .config("spark.sql.execution.arrow.enabled", "true")
        # .config("spark.driver.host", "127.0.0.1")
        # .config("spark.driver.bindAddress", "127.0.0.1")
        # .config("spark.executor.x509proxyname", "x509_u12409")
        # .config("spark.executor.memory", "1g")
        .config("spark.driver.memory", "16g")  # fixes java memory errors
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", 100000)
    )

    parameters['spark'] = _spark_initialize(
        config=spark_config, log_level="ERROR",
        spark_progress=False,
        laurelin_version="1.0.0"
    )

    datasets_mc = []
    datasets_data = []
    for group, samples in smp.items():
        for sample in samples:
            # if sample != 'data_B':
            # if sample != 'dy_m105_160_amc':
            if sample != 'vbf_powheg_dipole':
                continue
            if group == 'data':
                datasets_data.append(sample)
            else:
                datasets_mc.append(sample)

    timings = {}

    to_process = {
        'MC': datasets_mc,
        'DATA': datasets_data
    }
    for lbl, datasets in to_process.items():
        if len(datasets) == 0:
            continue
        print(f'Processing {lbl}')
        arg_sets = []
        for d in datasets:
            arg_sets.append({'dataset': d})

        tick1 = time.time()
        parameters['samp_infos'] = load_samples(datasets, parameters)
        timings[f'load {lbl}'] = time.time() - tick1

        tick2 = time.time()
        out = submit_job({}, parameters)
        timings[f'process {lbl}'] = time.time() - tick2

        print(out)

    elapsed = round(time.time() - tick, 3)
    print(f'Finished everything in {elapsed} s.')
    print('Timing breakdown:')
    print(timings)
