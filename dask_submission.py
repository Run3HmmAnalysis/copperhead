import time
import os, sys
import copy
import argparse
import socket
import math

from functools import partial
import traceback

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
from dask.distributed import get_client, get_worker, wait, as_completed, Worker, worker_client, secede, rejoin, fire_and_forget
dask.config.set({"temporary-directory": "/depot/cms/hmm/dask-temp/"})

import pandas as pd
import numpy as np
import pickle, lz4
import _pickle as pkl
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", dest="year", default='2016', action='store') # specify 2016, 2017 or 2018
parser.add_argument("-l", "--label", dest="label", default="test", action='store') # unique label for processing run

parser.add_argument("-ch", "--chunksize", dest="chunksize", default=10000, action='store')
parser.add_argument("-mch", "--maxchunks", dest="maxchunks", default=-1, action='store')

args = parser.parse_args()


#################### User settings: ####################

slurm_cluster_ip = '128.211.149.133:36634'

use_dask_outer = True # outer loop (over datasets, syst.variations, etc)
use_dask_inner = True # inner loop (over chunks of a dataset). Recommended: set to True unless you don't use Dask at all
max_outer_workers = 8 # limitation to prevent a situation where too many workers are blocked by outer tasks

parameters = {
    'year': args.year,
    'label': args.label,
    'global_out_path': '/depot/cms/hmm/coffea/',
    'out_path': f'{args.year}_{args.label}',
    'server': 'root://xrootd.rcac.purdue.edu/',
    'datasets_from': 'purdue',
#    'pt_variations': ['nominal', 'jer1_up', 'jer1_down'],
    'pt_variations': ['nominal'],
    'chunksize': int(args.chunksize),
    'maxchunks': None, # None to process all
    'save_output': True,
    'use_dask_outer': use_dask_outer,
    'use_dask_inner': use_dask_inner,
    'max_outer_workers': max_outer_workers,
    'slurm_cluster_ip': slurm_cluster_ip,
    'client': None,
}

parameters['out_dir'] = f"{parameters['global_out_path']}/{parameters['out_path']}/"



def load_sample(dataset, parameters):
    samp_info = SamplesInfo(year=parameters['year'], out_path=parameters['out_path'],\
                            server=parameters['server'], datasets_from='purdue', debug=False)
    samp_info.load(dataset, use_dask=parameters['use_dask_outer'])
    nevts = samp_info.finalize()
    return {dataset:samp_info}   

def submit_job(arg_set, parameters):
    dataset = arg_set['dataset']
    pt_variation = arg_set['pt_variation']
    samp_info = parameters['samp_infos'][dataset]
    if 'files' not in samp_info.fileset[dataset].keys():
        return 'Not loaded: '+dataset

    if parameters['use_dask_inner']:
        if parameters['use_dask_outer']:
            client = get_client()
        else:
            distributed = pytest.importorskip("distributed", minversion="1.28.1")
            distributed.config['distributed']['worker']['memory']['terminate'] = False
            client = distributed.Client(parameters['slurm_cluster_ip'])
        executor = dask_executor
        executor_args={'nano': True, 'client': client, 'workers_to_use': parameters["inner_workers"]}
    else:
        executor = iterative_executor
        executor_args={'nano': True}
    
    try:
        output = processor.run_uproot_job(samp_info.fileset, 'Events',
                                      DimuonProcessor(samp_info=samp_info,\
                                                      pt_variations=[pt_variation],\
                                                      do_btag_syst=False),
                                      executor, executor_args=executor_args,
                                      chunksize=parameters['chunksize'], maxchunks=parameters['maxchunks'])
    except Exception as e:
        tb = traceback.format_exc()
        return 'Failed: '+dataset+': '+str(e)+' '#+tb

    if parameters['save_output']:
        try:
            os.mkdir(parameters['out_dir'])
        except:
            pass
        for mode in output.keys():
            out_dir_ = f"{parameters['out_dir']}/{mode}/"
            out_path_ = f"{out_dir_}/{dataset}.coffea"
            try:
                os.mkdir(out_dir_)
            except:
                pass
            with lz4.frame.open(out_path_, 'wb') as fout:
                thepickle = pkl.dumps(output[mode])
                fout.write(thepickle)
#            util.save(output[mode], out_path_) # slower
            print(f"Saved output to {out_path_}")

    return 'Success: '+dataset
    
if __name__ == "__main__":
    if parameters['use_dask_outer'] and not parameters['use_dask_inner']:
        print("Parallelizing over datasets without parallelizing over chunks is really inefficient. Please reconsider.")
        sys.exit()

    t0 = time.time()
    smp = {
    'data':[
        'data_A',
        'data_B',
        'data_C',
        'data_D',
        'data_E',
        'data_F',
        'data_G',
        'data_H',
    ],
    'signal':[
        'ggh_amcPS',
        'vbf_powhegPS',
        'vbf_powheg_herwig',
        'vbf_powheg_dipole'
    ],
    'main_mc':[
        'dy_m105_160_amc',
        'dy_m105_160_vbf_amc',
        'ewk_lljj_mll105_160_py',
        'ewk_lljj_mll105_160_ptj0',
        'ewk_lljj_mll105_160_py_dipole',
        'ttjets_dl',
    ],
    'other_mc':[
        'ttjets_sl','ttz','ttw',
        'st_tw_top','st_tw_antitop',
        'ww_2l2nu','wz_2l2q',
        'wz_3lnu',
        'wz_1l1nu2q', 'zz',
    ],
    }

    # by default, include all samples
    # select the samples you want to process by skipping other samples
    datasets = []
    for group, samples in smp.items():
        for sample in samples:
#            if 'data' not in group: continue 
            if 'signal' not in group: continue
#            if 'main_mc' not in group: continue
#            if 'other_mc' not in group: continue
#            if 'vbf_powheg_dipole' not in sample: continue
            datasets.append(sample)

    arg_sets = []
    for d in datasets:
        for p in parameters['pt_variations']:
            arg_sets.append({'dataset':d, 'pt_variation':p})
    samp_infos = {}

    if parameters['use_dask_outer']:
        distributed = pytest.importorskip("distributed", minversion="1.28.1")
        distributed.config['distributed']['worker']['memory']['terminate'] = False
        client = distributed.Client(parameters['slurm_cluster_ip'])

        loaded = client.map(partial(load_sample, parameters=parameters),datasets) # list of futures
        loaded = client.gather(loaded) # list of results when futures are finished
        for l in loaded:
            samp_infos.update(l)
        parameters['samp_infos'] = samp_infos

        workers = list(client.scheduler_info()['workers'])
        if len(workers)<parameters['max_outer_workers']:
            print("Not enough workers to run everything in parallel. Aborting...")
            sys.exit()

        outer_workers = workers[0:min(len(arg_sets),parameters['max_outer_workers'])]
        parameters['all_workers'] = workers
        parameters['outer_workers'] = outer_workers
        parameters['inner_workers'] = [w for w in workers if w not in outer_workers]
        processed = client.map(partial(submit_job,parameters=parameters), arg_sets,\
                               key=[f'{a["dataset"]}, {a["pt_variation"]}' for a in arg_sets],\
                              resources={'processor': 1, 'reducer': 0}, workers=parameters['outer_workers'])

        for future in as_completed(processed):
            print(future.result())
    
    else:
        for d in datasets:
            samp_infos.update(load_sample(d,parameters))
        parameters['samp_infos'] = samp_infos        
        out = [submit_job(a, parameters) for a in arg_sets]
        for o in out:
            print(o)

    elapsed = time.time() - t0
    print(f'Finished in {elapsed}s.')
    




