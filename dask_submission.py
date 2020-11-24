import time
import os
import sys
import argparse
from functools import partial
import traceback

import lz4
import _pickle as pkl

import coffea
import coffea.processor as processor
from coffea.processor.executor import dask_executor, iterative_executor

from python.dimuon_processor import DimuonProcessor
from python.samples_info import SamplesInfo
import dask
from dask.distributed import Client, get_client, as_completed
dask.config.set({"temporary-directory": "/depot/cms/hmm/dask-temp/"})

print("Coffea version: ", coffea.__version__)

parser = argparse.ArgumentParser()
# specify 2016, 2017 or 2018
parser.add_argument("-y", "--year", dest="year",
                    default='2016', action='store')
# unique label for processing run
parser.add_argument("-l", "--label", dest="label",
                    default="test", action='store')
parser.add_argument("-ch", "--chunksize", dest="chunksize",
                    default=10000, action='store')
parser.add_argument("-mch", "--maxchunks", dest="maxchunks",
                    default=-1, action='store')

args = parser.parse_args()


# User settings:

slurm_cluster_ip = '128.211.149.133:38601'

# outer loop (over datasets, syst.variations, etc)
use_dask_outer = True

# inner loop (over chunks of a dataset).
# Recommended: set to True unless you don't use Dask at all
use_dask_inner = True

# limitation to prevent a situation where
# too many workers are blocked by outer tasks
max_outer_workers = 8

parameters = {
    'year': args.year,
    'label': args.label,
    'global_out_path': '/depot/cms/hmm/coffea/',
    'out_path': f'{args.year}_{args.label}',
    'server': 'root://xrootd.rcac.purdue.edu/',
    'datasets_from': 'purdue',
    # 'pt_variations': ['nominal', 'jer1_up', 'jer1_down'],
    'pt_variations': ['nominal'],
    'chunksize': int(args.chunksize),
    'maxchunks': None,  # None to process all
    'save_output': True,
    'use_dask_outer': use_dask_outer,
    'use_dask_inner': use_dask_inner,
    'max_outer_workers': max_outer_workers,
    'slurm_cluster_ip': slurm_cluster_ip,
    'client': None,
}

parameters['out_dir'] = f"{parameters['global_out_path']}/"\
                        f"{parameters['out_path']}/"


def load_sample(dataset, parameters):
    samp_info = SamplesInfo(year=parameters['year'],
                            out_path=parameters['out_path'],
                            server=parameters['server'],
                            datasets_from='purdue',
                            debug=False)
    samp_info.load(dataset, use_dask=parameters['use_dask_outer'])
    samp_info.finalize()
    return {dataset: samp_info}


def submit_job(arg_set, parameters):
    dataset = arg_set['dataset']
    pt_variation = arg_set['pt_variation']
    samp_info = parameters['samp_infos'][dataset]
    if 'files' not in samp_info.fileset[dataset].keys():
        return 'Not loaded: '+dataset

    if parameters['use_dask_inner']:
        if parameters['use_dask_outer']:
            client = get_client()
            workers_to_use = parameters["inner_workers"]
        else:
            client = Client(parameters['slurm_cluster_ip'])
            # client = dask.distributed.Client(
            #                    processes=True,
            #                    n_workers=10,
            #                    dashboard_address='128.211.149.133:34875',
            #                    threads_per_worker=1,
            #                    memory_limit='4GB')
            workers_to_use = list(client.scheduler_info()['workers'])
        executor = dask_executor
        executor_args = {
            'client': client,
            'schema': processor.NanoAODSchema,
            'workers_to_use': workers_to_use,
        }
    else:
        executor = iterative_executor
        executor_args = {'schema': processor.NanoAODSchema}

    try:
        output = processor.run_uproot_job(
                            samp_info.fileset,
                            'Events',
                            DimuonProcessor(samp_info=samp_info,
                                            pt_variations=[pt_variation],
                                            do_btag_syst=False,
                                            do_pdf=False),
                            executor,
                            executor_args=executor_args,
                            chunksize=parameters['chunksize'],
                            maxchunks=parameters['maxchunks'])
    except Exception as e:
        tb = traceback.format_exc()
        return 'Failed: '+dataset+': '+str(e)+' '+tb

    if parameters['save_output']:
        try:
            os.mkdir(parameters['out_dir'])
        except Exception as e:
            raise Exception(e)
        for mode in output.keys():
            out_dir_ = f"{parameters['out_dir']}/{mode}/"
            out_path_ = f"{out_dir_}/{dataset}.coffea"
            try:
                os.mkdir(out_dir_)
            except Exception as e:
                raise Exception(e)
            with lz4.frame.open(out_path_, 'wb') as fout:
                thepickle = pkl.dumps(output[mode])
                fout.write(thepickle)
            # util.save(output[mode], out_path_)  # slower
            print(f"Saved output to {out_path_}")

    return 'Success: '+dataset


if __name__ == "__main__":
    if parameters['use_dask_outer'] and not parameters['use_dask_inner']:
        print("Parallelizing over datasets without parallelizing"
              "over chunks is really inefficient. Please reconsider.")
        sys.exit()

    t0 = time.time()
    smp = {
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
            # 'ewk_lljj_mll105_160_py',
            # 'ewk_lljj_mll105_160_ptj0',
            'ewk_lljj_mll105_160_py_dipole',
            'ttjets_dl',
        ],
        'other_mc': [
            'ttjets_sl', 'ttz', 'ttw',
            'st_tw_top', 'st_tw_antitop',
            'ww_2l2nu', 'wz_2l2q',
            'wz_3lnu',
            'wz_1l1nu2q', 'zz',
        ],
    }

    # by default, include all samples
    # select the samples you want to process by skipping other samples
    datasets = []
    for group, samples in smp.items():
        for sample in samples:
            # if 'data' in group:
            # if ('main_mc' not in group) and ('signal' not in group):
            # if 'other_mc' not in group:
            # if 'vbf_powheg' not in sample:
            # if 'dy_m105_160_amc' not in sample:
            if 'signal' not in group:
                continue
            datasets.append(sample)

    arg_sets = []
    for d in datasets:
        for p in parameters['pt_variations']:
            arg_sets.append({'dataset': d, 'pt_variation': p})
    samp_infos = {}

    if parameters['use_dask_outer']:
        client = Client(parameters['slurm_cluster_ip'])
        # client = dask.distributed.Client(
        #                processes=True,
        #                n_workers=10,
        #                dashboard_address='128.211.149.133:34875',
        #                threads_per_worker=1,
        #                memory_limit='4GB')

        func = partial(load_sample, parameters=parameters)
        loaded = client.map(func, datasets)
        loaded = client.gather(loaded)

        for ld in loaded:
            samp_infos.update(ld)
        parameters['samp_infos'] = samp_infos

        workers = list(client.scheduler_info()['workers'])
        if len(workers) < parameters['max_outer_workers']:
            print("Not enough workers to run everything in parallel."
                  " Aborting...")
            sys.exit()

        outer_workers = workers[0:min(len(arg_sets),
                                parameters['max_outer_workers'])]
        inner_workers = [w for w in workers if w not in outer_workers]
        parameters['all_workers'] = workers
        parameters['outer_workers'] = outer_workers
        parameters['inner_workers'] = inner_workers
        key = [f'{a["dataset"]}, {a["pt_variation"]}' for a in arg_sets]
        processed = client.map(partial(submit_job, parameters=parameters),
                               arg_sets,
                               key=key,
                               resources={'processor': 1, 'reducer': 0},
                               workers=parameters['outer_workers'])
        for future in as_completed(processed):
            print(future.result())

    else:
        for d in datasets:
            samp_infos.update(load_sample(d, parameters))
        parameters['samp_infos'] = samp_infos
        out = [submit_job(a, parameters) for a in arg_sets]
        for o in out:
            print(o)

    elapsed = time.time() - t0
    print(f'Finished in {elapsed}s.')
