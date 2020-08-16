import time
import os, sys
import argparse
import socket

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

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", dest="year", default=2016, action='store') # specify 2016, 2017 or 2018
parser.add_argument("-l", "--label", dest="label", default="apr23", action='store') # unique label for processing run
parser.add_argument("-d", "--debug", action='store_true') # load only 1 file per dataset

# By default, a SLURM cluster will be created and jobs will be parallelized
# Alternative options:
parser.add_argument("-i", "--iterative", action='store_true') # run iteratively on 1 CPU
parser.add_argument("-loc", "--local", action='store_true') # create local Dask workers

args = parser.parse_args()


#################### User settings: ####################

server = 'root://xrootd.rcac.purdue.edu/'
global_out_path = '/depot/cms/hmm/coffea/'
slurm_cluster_ip = '128.211.149.140:46157'

# B-tag systematics significantly slow down 'nominal' processing 
# and they are only needed at the very last stage of the analysis.
# Let's keep them disabled for performance studies.
do_btag_syst = False

sample_sources = [
    'data',
    'main_mc',
    'signal',
    'other_mc',
]

smp = {}

smp['data'] = [
    'data_A',
    'data_B',
    'data_C',
    'data_D',
    'data_E',
    'data_F',
    'data_G',
    'data_H',
]

smp['main_mc'] = [
    'dy_m105_160_amc',
    'dy_m105_160_vbf_amc',
    'ewk_lljj_mll105_160_py',
    "ewk_lljj_mll105_160_ptj0",
    'ewk_lljj_mll105_160_py_dipole',
    'ttjets_dl',
]

smp['other_mc'] = [
    'ttjets_sl',
    'ttz','ttw',
    'st_tw_top','st_tw_antitop',
    'ww_2l2nu','wz_2l2q','wz_3lnu',
    'wz_1l1nu2q', 
    'zz',
]

smp['signal'] = [
    'ggh_amcPS',
    'vbf_powhegPS',
    'vbf_powheg_herwig',
    'vbf_powheg_dipole'
]


# Each JES/JER systematic variation (can be up/down) will run 
# approximately as long as the 'nominal' option.
# Therefore, processing all variations takes ~35 times longer than only 'nominal'.

pt_variations = []
pt_variations += ['nominal']
#pt_variations += ['Absolute', f'Absolute{args.year}']
#pt_variations += [f'Absolute{args.year}']
#pt_variations += ['BBEC1', f'BBEC1{args.year}']
#pt_variations += ['EC2', f'EC2{args.year}']
#pt_variations += ['FlavorQCD']
#pt_variations += ['HF',f'HF{args.year}']
#pt_variations += ['RelativeBal', f'RelativeSample{args.year}']
#pt_variations += ['jer1','jer2', 'jer3','jer4','jer5','jer6']

############################################################



samples = []
for sss in sample_sources:
    samples += smp[sss]

all_pt_variations = []
for ptvar in pt_variations:
    if ptvar=='nominal':
        all_pt_variations += ['nominal']
    else:
        all_pt_variations += [f'{ptvar}_up']
        all_pt_variations += [f'{ptvar}_down']    
    
if __name__ == "__main__":
    samp_info = SamplesInfo(year=args.year, out_path=f'{args.year}_{args.label}', server=server, datasets_from='purdue', debug=args.debug)

    if args.debug:
        samp_info.load(samples, parallelize_outer=1, parallelize_inner=1)
    else:
        samp_info.load(samples, parallelize_outer=46, parallelize_inner=1)

    samp_info.compute_lumi_weights()
    
    if args.iterative:
        tstart = time.time() 
        for variation in all_pt_variations:
            print(f"Jet pT variation: {variation}")
            output = processor.run_uproot_job(samp_info.full_fileset, 'Events',\
                                              DimuonProcessor(samp_info=samp_info, do_timer=True, pt_variations=[variation],\
                                                              debug=args.debug, do_btag_syst=do_btag_syst),\
                                          iterative_executor, executor_args={'nano': True})

        elapsed = time.time() - tstart
        print(f"Total time: {elapsed} s")
        sys.exit()
    
    if args.local:
        n_workers = 46
        distributed = pytest.importorskip("distributed", minversion="1.28.1")
        distributed.config['distributed']['worker']['memory']['terminate'] = False
        client = distributed.Client(processes=True, dashboard_address=None, n_workers=n_workers,\
                                    threads_per_worker=1, memory_limit='12GB') 
    else:
        distributed = pytest.importorskip("distributed", minversion="1.28.1")
        distributed.config['distributed']['worker']['memory']['terminate'] = False
        client = distributed.Client(slurm_cluster_ip)

        
    tstart = time.time()

    for variation in all_pt_variations:
        print(f"Jet pT variation: {variation}")
        for label, fileset in samp_info.filesets.items():
            # Not producing variated samples for minor backgrounds
            if (variation!='nominal') and not (('dy' in label) or ('ewk' in label) or ('vbf' in label) or ('ggh' in label) or ('ttjets_dl' in label)) or ('mg' in label):
                continue
            print(f"Processing: {label}, {variation}")
            output = processor.run_uproot_job(fileset, 'Events',\
                                              DimuonProcessor(samp_info=samp_info, pt_variations=[variation], do_btag_syst=do_btag_syst),\
                                              dask_executor, executor_args={'nano': True, 'client': client})

            out_dir = f"{global_out_path}/{samp_info.out_path}/"

            try:
                os.mkdir(out_dir)
            except:
                pass
    
            for mode in output.keys():
                out_dir_ = f"{out_dir}/{mode}/"
                out_path_ = f"{out_dir_}/{label}.coffea"
                try:
                    os.mkdir(out_dir_)
                except:
                    pass
                util.save(output[mode], out_path_)
            
            output.clear()
            print(f"Saved output to {out_dir}")
            
            del output
            
    elapsed = time.time() - tstart
    print(f"Total time: {elapsed} s")
