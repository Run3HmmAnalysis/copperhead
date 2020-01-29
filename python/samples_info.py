class SamplesInfo(object):
    def __init__(self, year='2016', out_path='/output/', at_purdue=True, debug=False):
        
        self.year = year
        self.out_path = out_path
        self.at_purdue = at_purdue
        self.debug = debug
        
        from config.datasets import datasets, lumi_data, all_dy, all_ewk
        self.server = 'root://xrootd.rcac.purdue.edu/'
            
        self.paths = datasets[self.year]
        self.datasets = datasets
        self.lumi_data = lumi_data
                
        self.lumi = 40000 # default value
        self.data_entries = 0

        self.samples = []
        self.missing_samples = []

        self.filesets = {}
        self.full_fileset = {}
        self.metadata = {}
        
        #--- Define regions and channels used in the analysis ---#
        self.regions = ['z-peak', 'h-sidebands', 'h-peak']
        self.channels = ['ggh_01j', 'ggh_2j', 'vbf']

        #--- Select samples for which unbinned data will be saved ---#
        self.data_samples = [s for s in self.samples if 'data' in s]
        self.mc_samples = [s for s in self.samples if 'data' not in s]
        self.signal_samples = ['ggh_amcPS', 'vbf_amcPS']
        self.main_bkg_samples = ['dy_m105_160_amc', 'dy_m105_160_vbf_amc',\
                                 'ttjets_dl', 'ewk_lljj_mll105_160', "ewk_lljj_mll105_160_ptj0"]
        self.datasets_to_save_unbin = self.data_samples + self.signal_samples + self.main_bkg_samples
        
        #--- Take overlapping samples and assign them to different regions ---#

        self.overlapping_samples = all_dy[self.year] + all_ewk[self.year]
        self.specific_samples = {
                'z-peak': {
                    'ggh_01j' : ["dy_0j", "dy_1j", "dy_2j", 'ewk_lljj_mll50_mjj120'],
                    'ggh_2j' : ["dy_0j", "dy_1j", "dy_2j", 'ewk_lljj_mll50_mjj120'],
                    'vbf' : ["dy_0j", "dy_1j", "dy_2j", 'ewk_lljj_mll50_mjj120']
                },
                'h-sidebands': {
                    'ggh_01j' : ['dy_m105_160_amc', 'ewk_lljj_mll105_160'],
                    'ggh_2j' : ['dy_m105_160_amc', 'ewk_lljj_mll105_160'],
                    'vbf' : ['dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160'],
                },
                'h-peak': {
                    'ggh_01j' : ['dy_m105_160_amc', 'ewk_lljj_mll105_160'],
                    'ggh_2j' : ['dy_m105_160_amc', 'ewk_lljj_mll105_160'],
                    'vbf' : ['dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160'],
                }
            }

        self.lumi_weights = {}

        
    def load(self, samples):
        import multiprocessing as mp
        import time
        t0 = time.time()

        pool = mp.Pool(mp.cpu_count())
        a = [pool.apply_async(self.load_sample, args=(s,)) for s in samples]
        results = []
        for process in a:
            process.wait()
            results.append(process.get())
        pool.close()

        for res in results: 
            sample = res['sample']
            if res['is_missing']:
                self.missing_samples.append(sample)
            else:
                self.samples.append(sample)
                self.filesets[sample] = {}
                self.filesets[sample][sample] = res['files']
                self.full_fileset[sample] = res['files']
    
                self.metadata[sample] = {}
                self.metadata[sample] = res['metadata']

                self.data_entries = self.data_entries + res['data_entries']
                

        if self.data_entries:
            print()   
            print(f"Loaded {self.data_entries} of {self.year} data events")
            self.lumi = lumi_data[self.year]['lumi']*self.data_entries/self.lumi_data[self.year]['events']
            prc = round(self.data_entries/self.lumi_data[self.year]['events']*100, 2)
            print(f"This is ~ {prc}% of {self.year} data.")
            print(f"Integrated luminosity {lumi}/pb")
            print()
        if self.missing_samples:
            print(f"Missing samples: {self.missing_samples}")

        t1 = time.time()        
        dt=round(t1-t0, 2)
        print(f"Loading took {dt} s")

    def load_sample(self, sample):
        import glob, tqdm
        import uproot
        from python.utils import read_via_xrootd
        print("Loading", sample)

        if sample not in self.paths:
            print(f"Couldn't load {sample}! Skipping.")
            return {'sample': sample, 'metadata': {}, 'files': {}, 'data_entries': 0, 'is_missing': True}
        
        all_files = []
        metadata = {}
        data_entries = 0
        
        if self.at_purdue:
            all_files = read_via_xrootd(self.paths[sample], self.server)
        else:
            all_files = [self.server+ f for f in glob.glob(self.paths[sample]+'*root')]      

        if self.debug:
            all_files = [all_files[0]]

        if 'data' in sample:
            for f in all_files:
                tree = uproot.open(f)['Events']
                data_entries += tree.numentries
        else:
            sumGenWgts = 0
            nGenEvts = 0
            for f in all_files:
                tree = uproot.open(f)['Runs']
                if 'NanoAODv6' in self.paths[sample]:
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
        return {'sample': sample, 'metadata': metadata, 'files': files, 'data_entries':data_entries, 'is_missing':False}

    def compute_lumi_weights(self):
        from config.cross_sections import cross_sections
        import json
        self.lumi_weights = {'data':1}
        for sample in self.mc_samples:
            N = self.metadata[sample]['sumGenWgts']
            if 'ewk_lljj_mll50_mjj120' in sample:
                xsec = cross_sections[sample]['2016']
            else:
                if 'ewk_lljj_mll105_160_ptj0' in sample:
                    xsec = cross_sections['ewk_lljj_mll105_160']            
                else:
                    xsec = cross_sections[sample]
            self.lumi_weights[sample] = xsec*self.lumi / N
