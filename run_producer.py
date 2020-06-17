import time
import os, sys
import argparse
import socket
import gc

import coffea
print("Coffea version: ", coffea.__version__)
from coffea import util
import coffea.processor as processor
from coffea.processor.executor import dask_executor
from coffea.processor.executor import iterative_executor

from python.dimuon_processor import DimuonProcessor
from python.samples_info import SamplesInfo

import pytest
import dask
from dask_jobqueue import SLURMCluster

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year", dest="year", default=2016, action='store')
parser.add_argument("-l", "--label", dest="label", default="apr23", action='store')
parser.add_argument("-d", "--debug", action='store_true')
parser.add_argument("-i", "--iterative", action='store_true')
parser.add_argument("-loc", "--local", action='store_true')
args = parser.parse_args()

sample_sources = [
    'data',
    'main_mc',
    'signal',
    'other_mc',
]

year = args.year

pt_variations = []
#pt_variations += ['nominal']
pt_variations += ['Absolute', f'Absolute{year}']
pt_variations += ['BBEC1', f'BBEC1{year}']
pt_variations += ['EC2', f'EC2{year}']
pt_variations += ['FlavorQCD']
pt_variations += ['HF',f'HF{year}']
pt_variations += ['RelativeBal', f'RelativeSample{year}']
pt_variations += ['jer1','jer2','jer3','jer4','jer5','jer6']

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
    'ewk_lljj_mll105_160', 
    'ewk_lljj_mll105_160_py',
    "ewk_lljj_mll105_160_ptj0",
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
    'vbf_amcPS',
    'vbf_powhegPS',
    'vbf_powheg_herwig',
    'vbf_powheg_dipole'
]

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
    samp_info = SamplesInfo(year=args.year, out_path=f'{args.year}_{args.label}', server='root://xrootd.rcac.purdue.edu/', datasets_from='purdue', debug=args.debug)

    if args.debug:
        samp_info.load(samples, parallelize_outer=1, parallelize_inner=1)
    else:
        samp_info.load(samples, parallelize_outer=46, parallelize_inner=1)

    samp_info.compute_lumi_weights()
    
    if args.iterative:
        tstart = time.time() 
        output = processor.run_uproot_job(samp_info.full_fileset, 'Events',\
                                          DimuonProcessor(samp_info=samp_info, do_timer=True, pt_variations=['Absolute_up'], debug=args.debug),\
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
        client = distributed.Client('128.211.149.133:36202')

        
    tstart = time.time()

    for variation in all_pt_variations:
        print(f"Jet pT variation: {variation}")
        for label, fileset in samp_info.filesets.items():
            # Not producing variated samples for minor backgrounds
            if (variation!='nominal') and not (('dy' in label) or ('ewk' in label) or ('vbf' in label) or ('ggh' in label)):
                continue
            print(f"Processing: {label}, {variation}")
            output = processor.run_uproot_job(fileset, 'Events',\
                                              DimuonProcessor(samp_info=samp_info, pt_variations=[variation]),\
                                              dask_executor, executor_args={'nano': True, 'client': client})

            out_dir = f"/depot/cms/hmm/coffea/{samp_info.out_path}/"

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
            gc.collect()
            
    elapsed = time.time() - tstart
    print(f"Total time: {elapsed} s")
