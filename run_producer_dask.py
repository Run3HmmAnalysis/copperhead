import time
import coffea
print("Coffea version: ", coffea.__version__)
import socket
import os
from coffea import util
import coffea.processor as processor
import pytest
from coffea.processor.executor import dask_executor
import dask
from python.dimuon_processor import DimuonProcessor
from python.samples_info import SamplesInfo
samples = [
## Data: ##
#    'data_A','data_B','data_C',
#    'data_D','data_E',
#    'data_F',
#    'data_G',
#    'data_H',
##

## MC for DNN training (Purdue): ##
    'dy_0j',
    'dy_1j',
    'dy_2j',
    'ewk_lljj_mll50_mjj120',
    
    'dy_m105_160_amc', 
    'dy_m105_160_vbf_amc',
    'ewk_lljj_mll105_160',
#    'ggh_amcPS', # GeoFit: get from Pisa
#    'vbf_amcPS', # GeoFit: get from Pisa
#    'ggh_powhegPS', # GeoFit: get from Pisa
#    'vbf_powhegPS', # GeoFit: get from Pisa

    'ttjets_dl', 
##
    
## MC for DNN training (Legnaro): ##    
#    "ewk_lljj_mll105_160_ptj0", # get from Pisa
##    
    

## Less important other MC: ##    
    'ttjets_sl',
#    'ttz', #missing for 2017
#    'ttw', #missing for 2017
    'st_tw_top','st_tw_antitop',
    'ww_2l2nu',
    'wz_2l2q',
    'wz_3lnu',
    'wz_1l1nu2q',
    'zz',
#    'www','wwz', #missing for 2017
#    'wzz','zzz', #missing for 2017
#    

# ##

]

purdue = 'root://xrootd.rcac.purdue.edu/'
legrano = 'root://t2-xrdcms.lnl.infn.it:7070//'

if __name__ == "__main__":
    suff="mar2"
#    samp_info = SamplesInfo(year="2016", out_path=f'all_2016_{suff}', server=purdue, datasets_from='purdue', debug=False)
#    samp_info = SamplesInfo(year="2016", out_path=f'all_2016_{suff}', server=legrano, datasets_from='pisa', debug=False)

    samp_info = SamplesInfo(year="2017", out_path=f'all_2017_{suff}', server=purdue, datasets_from='purdue', debug=False)
#    samp_info = SamplesInfo(year="2017", out_path=f'all_2017_{suff}', server=legrano, datasets_from='pisa', debug=False)

#    samp_info = SamplesInfo(year="2018", out_path=f'all_2018_{suff}', server=purdue, datasets_from='purdue', debug=False)
#    samp_info = SamplesInfo(year="2018", out_path=f'all_2018_{suff}', server=legrano, datasets_from='pisa', debug=False)

    samp_info.load(samples, nchunks=1, parallelize_outer=1, parallelize_inner=42)
    samp_info.compute_lumi_weights()

    n_workers = 24

    distributed = pytest.importorskip("distributed", minversion="1.28.1")
    distributed.config['distributed']['worker']['memory']['terminate'] = False
    client = distributed.Client(processes=True, dashboard_address=None, n_workers=n_workers, threads_per_worker=1) 

    tstart = time.time()
    from coffea.processor.executor import iterative_executor
    
    for label, fileset_ in samp_info.filesets_chunked.items():
        for ichunk, ifileset in enumerate(fileset_):
            print(f"Processing {label}, chunk {ichunk+1}/{samp_info.nchunks} ...")
            output = processor.run_uproot_job(ifileset, 'Events',\
                                      DimuonProcessor(samp_info=samp_info,\
                                                      do_fsr=True,\
                                                      do_roccor=True,\
                                                      evaluate_dnn=False, save_unbin=True,
                                                      do_lheweights=False, apply_jec=True, do_jer=True
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
                print("Output paths already exist")
            out_path_binned = f"{out_dir_binned}/{prefix}{label}_{ichunk}.coffea"
            out_path_unbinned = f"{out_dir_unbinned}/{prefix}{label}_{ichunk}.coffea"
            util.save(output['binned'], out_path_binned)
            util.save(output['unbinned'], out_path_unbinned)
            print(f"Saved output to {out_path_binned} and {out_path_unbinned}")   
    
    elapsed = time.time() - tstart
    print(f"Total time: {elapsed} s")
