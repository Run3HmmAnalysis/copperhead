import time
import json
import subprocess
import glob, tqdm
import numpy as np
import multiprocessing as mp
import pytest
import dask
from dask_jobqueue import SLURMCluster

import uproot
from coffea.lumi_tools import LumiData, LumiList, LumiMask

from config.parameters import parameters
from config.cross_sections import cross_sections

def read_via_xrootd(server, path):
    command = f"xrdfs {server} ls -R {path} | grep '.root'"
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    result = proc.stdout.readlines()
    if proc.stderr.readlines():
        print("Loading error! Check VOMS proxy.")
    result = [server + r.rstrip().decode("utf-8") for r in result]
    return result

class SamplesInfo(object):
    def __init__(self, year, out_path='/output/', xrootd=True, server='root://xrootd.rcac.purdue.edu/', datasets_from='purdue', debug=False):

        self.year = year
        self.out_path = out_path
        self.xrootd = xrootd
        self.debug = debug

        self.parameters = {k:v[self.year] for k,v in parameters.items()}

        self.is_mc = True
        
        if 'purdue' in datasets_from:
            from config.datasets import datasets, lumi_data
        elif 'pisa' in datasets_from:
            from config.datasets_pisa import datasets, lumi_data

        self.server = server

        self.paths = datasets[self.year]
        self.lumi_data = lumi_data

        if '2016' in self.year:
            self.lumi = 35900.
        elif '2017' in self.year:
            self.lumi = 41530.
        elif '2018' in self.year:
            self.lumi = 59970.
        #print('year: ', self.year)  
        #print('Default lumi: ', self.lumi)  

        self.data_entries = 0
        self.sample = ''
        self.samples = []

        self.fileset = {}
        self.metadata = {}

        #--- Define regions and channels used in the analysis ---#
        self.regions = ['z-peak', 'h-sidebands', 'h-peak']
        #self.channels = ['ggh_01j', 'ggh_2j', 'vbf']
        self.channels = ['vbf','vbf_01j','vbf_2j']

        self.lumi_weights = {}


    def load(self, sample, use_dask, client=None):
        t0 = time.time()
        
        if 'data' in sample:
            self.is_mc = False
        
        res = self.load_sample(sample, use_dask, client)
        
        self.sample = sample
        self.samples = [sample]
        self.fileset = {sample: res['files']}

        self.metadata = res['metadata']
        self.data_entries = res['data_entries']

        if self.data_entries:
            print()
            data_entries_total = self.lumi_data[self.year]['events']
            print(f"Total events in {self.year}: {data_entries_total}")

            print(f"Loaded {self.data_entries} of {self.year} data events")
            prc = round(self.data_entries/data_entries_total*100, 2)
            print(f"This is ~ {prc}% of {self.year} data.")

            print(f"Integrated luminosity: {self.lumi}/pb")
            print()
            

        t1 = time.time()
        dt=round(t1-t0, 2)
        print(f"Loading took {dt} s")

    def load_sample(self, sample, use_dask=False, client=None):
        if sample not in self.paths:
            print(f"Couldn't load {sample}! Skipping.")
            return {'sample': sample, 'metadata': {}, 'files': {}, 'data_entries': 0, 'is_missing': True}

        all_files = []
        metadata = {}
        data_entries = 0
        
        if self.xrootd:
            all_files = read_via_xrootd(self.server, self.paths[sample])
        else:
            all_files = [server+f for f in glob.glob(self.paths[sample]+'/**/**/*.root')]

        if self.debug:
            all_files = [all_files[0]]

        print(f"Loading {sample}: {len(all_files)} files")

        sumGenWgts = 0
        nGenEvts = 0
            
        if use_dask:
            from dask.distributed import get_client
            if not client:
                client = get_client()
            if 'data' in sample:
                work = client.map(self.get_data, all_files, priority=100)
            else:
                work = client.map(self.get_mc, all_files, priority=100)
            for w in work:
                ret = w.result()
                if 'data' in sample:
                    data_entries += ret['data_entries']
                else:
                    sumGenWgts += ret['sumGenWgts']
                    nGenEvts += ret['nGenEvts']            
        else:
            for f in all_files:
                if 'data' in sample:
                    tree = uproot.open(f)['Events']
                    data_entries += tree.numentries
                else:
                    tree = uproot.open(f)['Runs']
                    if ('NanoAODv6' in self.paths[sample]) or ('NANOV10' in self.paths[sample]):
                        sumGenWgts += tree.array('genEventSumw_')[0]
                        nGenEvts += tree.array('genEventCount_')[0]
                    else:
                        sumGenWgts += tree.array('genEventSumw')[0]
                        nGenEvts += tree.array('genEventCount')[0]
        metadata['sumGenWgts'] = sumGenWgts
        metadata['nGenEvts'] = nGenEvts

        files = {
            'files': all_files,
            'treename': 'Events'
        }
        return {'sample': sample, 'metadata': metadata, 'files': files,\
                'data_entries':data_entries, 'is_missing':False}

    def get_data(self, f):
        ret = {}
        file = uproot.open(f)
        tree = file['Events']
        ret['data_entries'] = tree.numentries
        return ret

    def get_mc(self, f):
        ret = {}
        tree = uproot.open(f)['Runs']
        if ('NanoAODv6' in f) or ('NANOV10' in f):
            ret['sumGenWgts'] = tree.array('genEventSumw_')[0]
            ret['nGenEvts'] = tree.array('genEventCount_')[0]
        else:
            ret['sumGenWgts'] = tree.array('genEventSumw')[0]
            ret['nGenEvts'] = tree.array('genEventCount')[0]
        return ret

    
    def finalize(self):
        if self.is_mc:
            N = self.metadata['sumGenWgts']
            numevents = self.metadata['nGenEvts']
            if isinstance(cross_sections[self.sample], dict):
                xsec = cross_sections[self.sample][self.year]
            else:
                xsec = cross_sections[self.sample]
            if N>0:
                self.lumi_weights[self.sample] = xsec*self.lumi / N
            else:
                self.lumi_weights[self.sample] = 0
            print(f"{self.sample}: events={numevents}")
            return numevents
        else:
            return self.data_entries
