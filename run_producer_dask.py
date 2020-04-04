import time
import coffea
print("Coffea version: ", coffea.__version__)
import socket
import os
import gc
from coffea import util
import coffea.processor as processor
import pytest
from coffea.processor.executor import dask_executor
import dask
from python.dimuon_processor import DimuonProcessor
from python.samples_info import SamplesInfo

sample_sources = [
    'data',
    'main_mc',
    'other_mc',
    'more_signal',
#    'pisa_mc'
]

year = '2016'
suff="apr2"
server_name = 'purdue'
#server_name = 'legnaro'
try_cluster = True
do_jec=True
do_jer=False
do_roch=True
do_fsr=True
do_geofit=False
evaluate_dnn = False
do_jecunc=False
do_pdf=False
debug=False
#if not do_jec:
#    suff += '_nojec'
if (not do_roch) and (not do_fsr):
    suff += '_raw'
elif not do_roch:
    suff += '_noroch'
if do_jecunc:
    suff+= '_jecunc'

smp = {}

smp['data'] = [
    'data_A',
    'data_B',
    'data_C',
    'data_D','data_E',
    'data_F',
    'data_G',
    'data_H',
]

smp['main_mc'] = [
    'dy_0j',
    'dy_1j',
    'dy_2j',
    'ewk_lljj_mll50_mjj120',
    'dy_m105_160_amc',
    'dy_m105_160_vbf_amc',
    'ewk_lljj_mll105_160',
    'ttjets_dl',
]

smp['pisa_mc'] = [
        'ggh_amcPS', # GeoFit: get from Pisa
        'vbf_amcPS', # GeoFit: get from Pisa
        'ggh_powhegPS', # GeoFit: get from Pisa
        'vbf_powhegPS', # GeoFit: get from Pisa
        "ewk_lljj_mll105_160_ptj0",
]

smp['other_mc'] = [
    'ttjets_sl',
    'ttz', #missing for 2017
    'ttw', #missing for 2017
    'st_tw_top','st_tw_antitop',
    'ww_2l2nu',
    'wz_2l2q',
    'wz_3lnu',
    #    'wz_1l1nu2q', # broken for 2016,2017
    'zz',
    'www','wwz', #missing for 2017
    'wzz','zzz', #missing for 2017
]

smp['more_signal'] = [
#    'ggh_amcPS',
#    'vbf_amcPS',
#    'vbf_powhegPS',
#    'vbf_powheg_herwig',
]

samples = []
for sss in sample_sources:
    samples += smp[sss]

if server_name=='purdue':
    server = 'root://xrootd.rcac.purdue.edu/'
    datasets_from='purdue'
elif server_name=='legnaro':
    server = 'root://t2-xrdcms.lnl.infn.it:7070//'
    datasets_from='pisa'

if __name__ == "__main__":
    samp_info = SamplesInfo(year=year, out_path=f'all_{year}_{suff}', server=server, datasets_from=datasets_from, debug=debug)

    if debug:
        samp_info.load(samples, nchunks=1, parallelize_outer=32, parallelize_inner=1)
    else:
        if 'data' in sample_sources:
            samp_info.load(samples, nchunks=1, parallelize_outer=1, parallelize_inner=46)
        else:
            samp_info.load(samples, nchunks=1, parallelize_outer=3, parallelize_inner=13)

    samp_info.compute_lumi_weights()

    if try_cluster:
        from dask_jobqueue import SLURMCluster
#        cluster = SLURMCluster( project='cms-express', cores=1, memory='16GB', walltime='23:59:00',\
#                                job_extra=['--qos=normal', '-o dask_logs/dask_job.%j.%N.out','-e dask_logs/dask_job.%j.%N.error'])
#        cluster.scale(300)
        distributed = pytest.importorskip("distributed", minversion="1.28.1")
        distributed.config['distributed']['worker']['memory']['terminate'] = False
#        client = distributed.Client(cluster)
        client = distributed.Client('128.211.149.133:36152')
    
    else:
        n_workers = 48
        distributed = pytest.importorskip("distributed", minversion="1.28.1")
        distributed.config['distributed']['worker']['memory']['terminate'] = False
        client = distributed.Client(processes=True, dashboard_address=None, n_workers=n_workers, threads_per_worker=1, memory_limit='12GB') 

    tstart = time.time()
    from coffea.processor.executor import iterative_executor
    
    for label, fileset_ in samp_info.filesets_chunked.items():
        for ichunk, ifileset in enumerate(fileset_):
            print(f"Processing {label}, chunk {ichunk+1}/{samp_info.nchunks} ...")
            output = processor.run_uproot_job(ifileset, 'Events',\
                                      DimuonProcessor(samp_info=samp_info,\
                                                      do_fsr=do_fsr,\
                                                      do_roch=do_roch,\
                                                      evaluate_dnn=evaluate_dnn, save_unbin=True,
                                                      do_lheweights=False, do_jec=do_jec, do_jer=do_jer, do_nnlops=True,
                                                      do_geofit=do_geofit, do_jecunc=do_jecunc, do_pdf=do_pdf, do_btagsf=True,
                                                  ),\
                                      dask_executor,\
                                      executor_args={'nano': True, 'client': client})


            prefix = ""
            out_dir = f"/depot/cms/hmm/coffea/{samp_info.out_path}/"
            out_dir_binned = f"{out_dir}/binned/"
            out_dir_unbinned = f"{out_dir}/unbinned/"
            try:
                os.mkdir(out_dir)
                os.mkdir(out_dir_binned)
                os.mkdir(out_dir_unbinned)
            except:
                pass
#                print("Output paths already exist")
            out_path_binned = f"{out_dir_binned}/{prefix}{label}_{ichunk}.coffea"
            out_path_unbinned = f"{out_dir_unbinned}/{prefix}{label}_{ichunk}.coffea"
            util.save(output['binned'], out_path_binned)
            util.save(output['unbinned'], out_path_unbinned)
            for c in ['vbf']:
                for r in ['z-peak', 'h-sidebands', 'h-peak']:
                    path = out_dir+f'/event_weight_{c}_{r}/'
                    try:
                        os.mkdir(path)
                    except:
                        pass
                    util.save(output[f'event_weight_{c}_{r}'], path+f'{prefix}{label}_{ichunk}.coffea')
            
            output.clear()
            print(f"Saved output to {out_dir}")
            
            del output
            gc.collect()
            
    elapsed = time.time() - tstart
    print(f"Total time: {elapsed} s")
