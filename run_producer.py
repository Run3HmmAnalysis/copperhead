import time
import os, sys
import argparse
import socket
import math

import coffea
print("Coffea version: ", coffea.__version__)
from coffea import util
import coffea.processor as processor
from coffea.processor.executor import dask_executor, iterative_executor

from python.dimuon_processor import DimuonProcessor
from python.samples_info import SamplesInfo

import pytest
import dask
from dask_jobqueue import SLURMCluster

import pandas as pd
import numpy as np
import pickle
import datetime

parser = argparse.ArgumentParser()
# specify 2016, 2017 or 2018
parser.add_argument("-y", "--year", dest="year",
                    default='2016', action='store')
# unique label for processing run
parser.add_argument("-l", "--label", dest="label",
                    default="test", action='store')
parser.add_argument("-d", "--debug", action='store_true')
# load only 1 file per dataset

# Processing options:
# run on Slurm cluster using Dask
parser.add_argument("-dask", "--dask", action='store_true')
# run iteratively on 1 CPU
parser.add_argument("-i", "--iterative", action='store_true')
# create local Dask workers
parser.add_argument("-dl", "--dasklocal", action='store_true')
# run with Spark
parser.add_argument("-s", "--spark", action='store_true')
parser.add_argument("-ch", "--chunksize", dest="chunksize", default=100000, action='store')
parser.add_argument("-mch", "--maxchunks", dest="maxchunks",
                    default=-1, action='store')

args = parser.parse_args()

# User settings:

server = 'root://xrootd.rcac.purdue.edu/'
global_out_path = '/depot/cms/hmm/coffea/'
slurm_cluster_ip = '128.211.149.140:36984'
chunksize = int(args.chunksize)  # default 100000
print(f"Running with chunksize {chunksize}")
if int(args.maxchunks) > 0:
    maxchunks = int(args.maxchunks)
    print(f"Will process {maxchunks} chunks")
else:
    maxchunks = None  # default None (process all chunks)
    print("Will process all chunks")

save_output = False
do_reduce = True
# if set to False, the merging step will not be performed in Dask executors, outputs will be saved unmerged

save_diagnostics = False
testing_db_path = '/depot/cms/hmm/performance_tests_db_fsrcache.pkl'

show_timer = True

# B-tag systematics significantly slow down 'nominal' processing
# and they are only needed at the very last stage of the analysis.
# Let's keep them disabled for performance studies.
do_btag_syst = False

sample_sources = [
    'data',
    # 'main_mc',
    # 'signal',
    # 'other_mc',
]

smp = {
    'data': [
        # 'data_A',
        # 'data_B',
        # 'data_C',
        # 'data_D',
        # 'data_E',
        # 'data_F',
        #'data_G',
         'data_H',
    ],
    'signal': [
        # 'ggh_amcPS',
        # 'vbf_powhegPS',
        # 'vbf_powheg_herwig',
        'vbf_powheg_dipole'
    ],
    'main_mc': [
        'dy_m105_160_amc',
        # 'dy_m105_160_vbf_amc',
        # 'ewk_lljj_mll105_160_py',
        # 'ewk_lljj_mll105_160_ptj0',
        # 'ewk_lljj_mll105_160_py_dipole',
        # 'ttjets_dl',
    ],
    'other_mc': [
        # 'ttjets_sl', 'ttz', 'ttw',
        # 'st_tw_top', 'st_tw_antitop',
        # 'ww_2l2nu', 'wz_2l2q',
        'wz_3lnu',
        # 'wz_1l1nu2q', 'zz',
    ],
}

# Each JES/JER systematic variation (can be up/down)
# will run approximately as long as the 'nominal' option.
# Therefore, processing all variations takes
# ~35 times longer than only 'nominal'.

pt_variations = []
pt_variations += ['nominal']
# pt_variations += ['Absolute', f'Absolute{args.year}']
# pt_variations += [f'Absolute{args.year}']
# pt_variations += ['BBEC1', f'BBEC1{args.year}']
# pt_variations += ['EC2', f'EC2{args.year}']
# pt_variations += ['FlavorQCD']
# pt_variations += ['HF', f'HF{args.year}']
# pt_variations += ['RelativeBal', f'RelativeSample{args.year}']
# pt_variations += ['jer1', 'jer2', 'jer3', 'jer4', 'jer5', 'jer6']

if args.iterative:
    method = 'Iterative'
elif args.spark:
    method = 'Spark'
elif args.dask:
    method = 'Dask+Slurm'
elif args.dasklocal:
    method = 'DaskLocal'
else:
    raise Exception("Running method not specified!")
    sys.exit()


if __name__ == "__main__":

    t_start = time.time()

    # Prepare resources

    t_prep_start = time.time()
    
    samples = []
    for sss in sample_sources:
        samples += smp[sss]
    
    dataset = samples[0] 
    
    samp_info = SamplesInfo(year=args.year, 
                            out_path=f'{args.year}_{args.label}', 
                            server=server,
                            datasets_from='purdue',
                            debug=args.debug)
    samp_info.load(dataset, use_dask=False)
    
    nevts = samp_info.finalize()
    
    nevts_all = samp_info.compute_lumi_weights()

    nevts = 0
    nchunks = 0
    for k, v in nevts_all.items():
        nevts += v
        nchunks += math.ceil(v / chunksize)
    if (nchunks > int(args.maxchunks)) & (int(args.maxchunks) > 0):
        nchunks = int(args.maxchunks)
        nevts = nchunks*chunksize

    all_pt_variations = []
    for ptvar in pt_variations:
        if ptvar=='nominal':
            all_pt_variations += ['nominal']
        else:
            all_pt_variations += [f'{ptvar}_up']
            all_pt_variations += [f'{ptvar}_down']

    out_dir = f"{global_out_path}/{samp_info.out_path}/"
    try:
        os.mkdir(out_dir)
    except Exception as e:
        raise Exception(e)

    # Initialize clusters for processing

    if method == 'Iterative':
        nworkers = 1

    elif method == 'Spark':
        import pyspark.sql
        from pyarrow.compat import guid
        from coffea.processor.spark.detail import _spark_initialize, _spark_stop
        from coffea.processor.spark.spark_executor import spark_executor
        spark_config = pyspark.sql.SparkSession.builder \
                        .appName('spark-executor-test-%s' % guid()) \
                        .master('local[*]') \
                        .config('spark.driver.memory', '16g') \
                        .config('spark.executor.memory', '16g') \
                        .config('spark.sql.execution.arrow.enabled','true') \
                        .config('spark.sql.execution.arrow.maxRecordsPerBatch', chunksize)

        spark = _spark_initialize(config=spark_config, log_level='WARN', 
                                  spark_progress=False, laurelin_version='1.0.0')
        thread_workers = 2
        nworkers = 1  # placeholder
        print(spark)

    elif method=='Dask+Slurm':
        flexible_workers = False
        distributed = pytest.importorskip("distributed", minversion="1.28.1")
        distributed.config['distributed']['worker']['memory']['terminate'] = False
        client = distributed.Client(slurm_cluster_ip)
        nworkers = len(client._scheduler_identity.get("workers", {}))
        print(f"Starting processing with {nworkers} workers")

    elif method=='DaskLocal':
        target_nworkers = 46
        distributed = pytest.importorskip("distributed", minversion="1.28.1")
        distributed.config['distributed']['worker']['memory']['terminate'] = False
        client = distributed.Client(processes=True, dashboard_address=None, n_workers=target_nworkers,
                                threads_per_worker=1, memory_limit='12GB') 
        nworkers = len(client._scheduler_identity.get("workers", {}))
        print(f"Starting processing with {nworkers} workers")

    t_prep = time.time() - t_prep_start

    # Start processing

    t_run_start = time.time()

    for variation in all_pt_variations:
        for sample_name in samples:
            fileset = samp_info.fileset
#        for sample_name, fileset in samp_info.filesets.items():
            if (variation!='nominal') and\
                not(('dy' in sample_name) or\
                    ('ewk' in sample_name) or\
                    ('vbf' in sample_name) or\
                    ('ggh' in sample_name) or\
                    ('ttjets_dl' in sample_name)) or\
                    ('mg' in sample_name):
                continue
            print(f"Processing: {sample_name}, {variation}")

            if method == 'Iterative':
                output, metrics = processor.run_uproot_job(
                                    fileset,
                                    'Events',
                                    DimuonProcessor(
                                        samp_info=samp_info,
                                        do_timer=show_timer,
                                        pt_variations=[variation],
                                        debug=args.debug,
                                        do_btag_syst=do_btag_syst),
                                    iterative_executor, 
                                    executor_args={
                                        'nano': True,
                                        'savemetrics':True}, 
                                    chunksize=chunksize,
                                    maxchunks=maxchunks)
            elif method == 'Spark':
                output  = processor.run_spark_job(
                                    fileset, 
                                    DimuonProcessor(
                                        samp_info=samp_info,
                                        pt_variations=[variation],
                                        do_btag_syst=do_btag_syst),
                                    spark_executor,
                                    spark=spark,
                                    partitionsize=chunksize,
                                    thread_workers=thread_workers,
                                    executor_args={
                                        'file_type': 'edu.vanderbilt.accre.laurelin.Root',
                                        'cache': False,
                                        'nano': True,
                                        'retries': 5,
                                        'laurelin_version': '1.0.0'})

            elif (method == 'Dask+Slurm') or (method == 'DaskLocal'):
                save_args = {
                    'save': save_output,
                    'out_dir': out_dir,
                    'label': sample_name
                }
                if not do_reduce:
                    save_output=False
                output = processor.run_uproot_job(
                    fileset,
                    'Events',
                    DimuonProcessor(samp_info=samp_info,
                                    pt_variations=[variation],
                                    do_btag_syst=do_btag_syst),
                    dask_executor, 
                    executor_args={'nano': True,
                                   'client': client,
                                   'do_reduce': do_reduce,
                                   'save_args': save_args},
                    chunksize=chunksize,
                    maxchunks=maxchunks)

    t_run = time.time() - t_start
    print(f"Total running time (with overhead): {t_run}")

    # Save outputs

    if save_output:
        t_save_start = time.time()
        for mode in output.keys():
            out_dir_ = f"{out_dir}/{mode}/"
            out_path_ = f"{out_dir_}/{sample_name}.coffea"
            try:
                os.mkdir(out_dir_)
            except Exception as e:
                raise Exception(e)
            util.save(output[mode], out_path_)
            print(f"Saved output to {out_dir}")
        t_save = time.time() - t_save_start
    else:
        t_save = 0

t = time.time() - t_start

if save_diagnostics:
    row = {
        'datetime': datetime.datetime.now(),
        'method': method,
        'samples': np.asarray(samples),
        'nevts': nevts,
        'maxchunks': int(args.maxchunks),
        'chunksize': chunksize, 
        'nchunks': nchunks,
        'nworkers': nworkers,  # so far only for Dask
        'nvariations': len(all_pt_variations),
        'prep_time': t_prep,
        'run_time': t_run,
        'save_time': t_save,
        'total_time': t,
        'do_reduce': do_reduce
    }
    try:
        with open(testing_db_path, 'rb') as db_file:
            db = pickle.load(db_file)
        print(f"Loaded database from {testing_db_path} to save diagnostics")
        idx = len(db)
    except Exception as e:
        db = pd.DataFrame(columns=list(row.keys()), dtype=float)
        idx = 0
    for k, v in row.items():
        db.loc[idx, k] = v
    print(db)
    db.to_pickle(testing_db_path)
    print(f"Database saved to {testing_db_path}")
