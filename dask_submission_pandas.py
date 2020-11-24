import time
import os
import argparse
import traceback

import coffea.processor as processor
from python.executor import dask_executor, run_uproot_job
from python.dimuon_processor_pandas import DimuonProcessor
from python.samples_info import SamplesInfo

import dask
import dask.dataframe as dd
from dask.distributed import Client
dask.config.set({"temporary-directory": "/depot/cms/hmm/dask-temp/"})

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

local_cluster = True  # set to False for Slurm cluster
slurm_cluster_ip = '128.211.149.133:38601'

mch = None if int(args.maxchunks) < 0 else int(args.maxchunks)

pt_variations = []
pt_variations += ['nominal']
pt_variations += ['Absolute', f'Absolute{args.year}']
pt_variations += ['BBEC1', f'BBEC1{args.year}']
pt_variations += ['EC2', f'EC2{args.year}']
pt_variations += ['FlavorQCD']
pt_variations += ['HF', f'HF{args.year}']
pt_variations += [
    'RelativeBal',
    f'RelativeSample{args.year}'
]
pt_variations += ['jer1', 'jer2', 'jer3']
pt_variations += ['jer4', 'jer5', 'jer6']

all_pt_variations = []
for ptvar in pt_variations:
    if ptvar == 'nominal':
        all_pt_variations += ['nominal']
    else:
        all_pt_variations += [f'{ptvar}_up']
        all_pt_variations += [f'{ptvar}_down']

parameters = {
    'year': args.year,
    'label': args.label,
    'global_out_path': '/depot/cms/hmm/coffea/',
    'out_path': f'{args.year}_{args.label}',
    'server': 'root://xrootd.rcac.purdue.edu/',
    'datasets_from': 'purdue',
    # 'pt_variations': ['nominal', 'jer1_up', 'jer1_down'],
    # 'pt_variations': ['nominal', 'Absolute2016_up'],
    'pt_variations': ['nominal'],
    # 'pt_variations': all_pt_variations,
    'chunksize': int(args.chunksize),
    'maxchunks': mch,  # None to process all
    'save_output': True,
    'local_cluster': local_cluster,
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
    samp_info.load(dataset, use_dask=True, client=parameters['client'])
    samp_info.finalize()
    return {dataset: samp_info}


def load_samples(datasets, parameters):
    samp_info_total = SamplesInfo(year=parameters['year'],
                                  out_path=parameters['out_path'],
                                  server=parameters['server'],
                                  datasets_from='purdue',
                                  debug=False)
    for d in datasets:
        if d in samp_info_total.samples:
            continue
        si = load_sample(d, parameters)[d]
        if 'files' not in si.fileset[d].keys():
            continue
        samp_info_total.data_entries += si.data_entries
        samp_info_total.fileset.update(si.fileset)
        samp_info_total.metadata.update(si.metadata)
        samp_info_total.lumi_weights.update(si.lumi_weights)
        samp_info_total.samples.append(sample)
    return samp_info_total


def submit_job(arg_set, parameters):
    executor = dask_executor
    executor_args = {
        'client': parameters['client'],
        'schema': processor.NanoAODSchema,
        'compression': None
    }

    try:
        output = run_uproot_job(parameters['samp_infos'].fileset, 'Events',
                                DimuonProcessor(
                                    samp_info=parameters['samp_infos'],
                                    pt_variations=parameters['pt_variations'],
                                    do_btag_syst=False, do_pdf=False
                                ),
                                executor, executor_args=executor_args,
                                chunksize=parameters['chunksize'],
                                maxchunks=parameters['maxchunks'])
    except Exception as e:
        tb = traceback.format_exc()
        return 'Failed: '+str(e)+' '+tb

    output.columns = [' '.join(col).strip() for col in output.columns.values]
    print(output.compute())

    if parameters['save_output']:
        try:
            os.mkdir(parameters['out_dir'])
        except Exception:
            pass
        out_dir_ = f"{parameters['out_dir']}/nominal/"
        try:
            os.mkdir(out_dir_)
        except Exception:
            pass
        for ds in output.s.unique():
            out_path_ = f"{out_dir_}/{ds}/"
            dd.to_parquet(df=output[output.s == ds], path=out_path_)
            print(f"Saved output to {out_path_}")
    return 'Success!'


if __name__ == "__main__":
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
            # 'ggh_amcPS',
            # 'vbf_powhegPS',
            # 'vbf_powheg_herwig',
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
            # if 'data' in group: continue
            if 'signal' not in group:
                continue
            # if ('main_mc' not in group) and ('signal' not in group): continue
            # if 'main_mc' not in group: continue
            # if 'other_mc'  in group: continue
            # if 'vbf_powheg' not in sample: continue
            # if 'dy_m105_160_amc' not in sample: continue
            datasets.append(sample)

    arg_sets = []
    for d in datasets:
        arg_sets.append({'dataset': d})

    if parameters['local_cluster']:
        parameters['client'] = dask.distributed.Client(
                                    processes=True,
                                    n_workers=40,
                                    dashboard_address='128.211.149.133:34875',
                                    threads_per_worker=1,
                                    memory_limit='4GB')
    else:
        parameters['client'] = Client(parameters['slurm_cluster_ip'])

    parameters['samp_infos'] = load_samples(datasets, parameters)
    out = submit_job({}, parameters)
    print(out)
    elapsed = time.time() - t0
    print(f'Finished in {elapsed}s.')
