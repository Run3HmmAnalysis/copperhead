from coffea.lumi_tools import LumiData, LumiList, LumiMask

def read_via_xrootd(server, path):
    import subprocess
    command = f"xrdfs {server} ls -R {path} | grep '.root'"
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    result = proc.stdout.readlines()
    if proc.stderr.readlines():
        print(proc.stderr.readlines())
#        print(result)
    result = [server + r.rstrip().decode("utf-8") for r in result]
    return result

class SamplesInfo(object):
    def __init__(self, year, out_path='/output/', at_purdue=True, server='root://xrootd.rcac.purdue.edu/', datasets_from='purdue', debug=False, example=False, example_datasets=[]):

        self.year = year
        self.out_path = out_path
        self.at_purdue = at_purdue
        self.debug = debug

        from config.parameters import parameters
        self.parameters = {k:v[self.year] for k,v in parameters.items()}
        
        if example:
            datasets = example_datasets
            from config.datasets import lumi_data, all_dy, all_ewk
        elif 'purdue' in datasets_from:
            from config.datasets import datasets, lumi_data, all_dy, all_ewk
        elif 'pisa' in datasets_from:
            from config.datasets_pisa import datasets, lumi_data, all_dy, all_ewk

        self.server = server

        self.paths = datasets[self.year]
        self.datasets = datasets
        self.lumi_data = lumi_data

        if '2016' in self.year:
#            self.lumi = 35917.15021920668
            self.lumi = 35900.
        elif '2017' in self.year:
#            self.lumi = 41525.06046688122
            self.lumi = 41530.
        elif '2018' in self.year:
#            self.lumi = 59725.42030414335  
            self.lumi = 59970.
        print('year: ', self.year)  
        print('Default lumi: ', self.lumi)  
        self.data_entries = 0

#        self.lumi_mask = LumiMask(self.parameters['lumimask'])
        self.lumi_list = LumiList()
    
        self.samples = []
        self.missing_samples = []

        self.filesets = {}
        self.full_fileset = {}
        self.metadata = {}

        #--- Define regions and channels used in the analysis ---#
        self.regions = ['z-peak', 'h-sidebands', 'h-peak']
        #self.channels = ['ggh_01j', 'ggh_2j', 'vbf']
        self.channels = ['vbf','vbf_01j','vbf_2j']
        
        #--- Select samples for which unbinned data will be saved ---#
        self.signal_samples = ['ggh_amcPS', 'vbf_amcPS']
        self.additional_signal = ['ggh_powheg', 'ggh_powhegPS', 'vbf_powheg', 'vbf_powhegPS', 'vbf_powheg_herwig']
        self.main_bkg_samples = ['dy_m105_160_amc', 'dy_m105_160_vbf_amc',\
                                 'ttjets_dl', 'ewk_lljj_mll105_160', "ewk_lljj_mll105_160_ptj0"]
        self.datasets_to_save_unbin = self.signal_samples + self.additional_signal + self.main_bkg_samples

        #--- Take overlapping samples and assign them to different regions ---#

        self.overlapping_samples = all_dy[self.year] + all_ewk[self.year]
        self.specific_samples = {
                'z-peak': {
                    'ggh_01j' : ["dy_0j", "dy_1j", "dy_2j", 'ewk_lljj_mll50_mjj120'],
                    'ggh_2j' : ["dy_0j", "dy_1j", "dy_2j", 'ewk_lljj_mll50_mjj120'],
                    'vbf' : ["dy_0j", "dy_1j", "dy_2j", 'ewk_lljj_mll50_mjj120'],
                    'vbf_01j' : ["dy_0j", "dy_1j", "dy_2j", 'ewk_lljj_mll50_mjj120'],
                    'vbf_2j' : ["dy_0j", "dy_1j", "dy_2j", 'ewk_lljj_mll50_mjj120']
                },
                'h-sidebands': {
                    'ggh_01j' : ['dy_m105_160_amc', 'ewk_lljj_mll105_160','ewk_lljj_mll105_160_ptj0'],
                    'ggh_2j' : ['dy_m105_160_amc', 'ewk_lljj_mll105_160','ewk_lljj_mll105_160_ptj0'],
                    'vbf' : ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160','ewk_lljj_mll105_160_ptj0'],
                    'vbf_01j' : ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160','ewk_lljj_mll105_160_ptj0'],
                    'vbf_2j' : ['dy_m105_160_amc', 'dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160','ewk_lljj_mll105_160_ptj0'],
                },
                'h-peak': {
                    'ggh_01j' : ['dy_m105_160_amc', 'ewk_lljj_mll105_160','ewk_lljj_mll105_160_ptj0'],
                    'ggh_2j' : ['dy_m105_160_amc', 'ewk_lljj_mll105_160','ewk_lljj_mll105_160_ptj0'],
                    'vbf' : ['dy_m105_160_amc','dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160','ewk_lljj_mll105_160_ptj0'],
                    'vbf_01j' : ['dy_m105_160_amc','dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160','ewk_lljj_mll105_160_ptj0'],
                    'vbf_2j' : ['dy_m105_160_amc','dy_m105_160_vbf_amc', 'ewk_lljj_mll105_160','ewk_lljj_mll105_160_ptj0'],
                }
            }

        self.lumi_weights = {}


    def load(self, samples, nchunks=1, parallelize_outer=1, parallelize_inner=1):
        import multiprocessing as mp
        import time
        import numpy as np
        t0 = time.time()
        if (parallelize_outer*parallelize_inner)>(mp.cpu_count()-1):
            print(f"Trying to create too many workers ({parallelize_outer*parallelize_inner})! Max allowed: {mp.cpu_count()-1}.")
            raise
                
        self.nchunks = nchunks
        
        big_samples = [s for s in samples if ('data' in s) or ('dy' in s) or ('ewk' in s) or ('ttjets' in s)]
        small_samples = [s for s in samples if s not in big_samples]
        
        results = []
        for s in big_samples:
            results.append(self.load_sample(s, parallelize_outer))
            
        pool = mp.Pool(parallelize_outer)
        a = [pool.apply_async(self.load_sample, args=(s,)) for s in small_samples]
        for process in a:
            process.wait()
            results.append(process.get())
            pool.close()

#        if parallelize_outer>1:
#            pool = mp.Pool(parallelize_outer)
#            a = [pool.apply_async(self.load_sample, args=(s,)) for s in samples]
#            results = []
#            for process in a:
#                process.wait()
#                results.append(process.get())
#                pool.close()
#        else:
#            results=[]
#            for s in samples:
#                results.append(self.load_sample(s, parallelize_inner))
                
                
        self.filesets_chunked = {}
        for res in results:
            sample = res['sample']
            if res['is_missing']:
                self.missing_samples.append(sample)
            else:
                self.samples.append(sample)
                self.filesets[sample] = {}
                self.filesets_chunked[sample] = []
                self.filesets[sample][sample] = res['files']
                self.full_fileset[sample] = res['files']

                self.metadata[sample] = {}
                self.metadata[sample] = res['metadata']
                self.data_entries = self.data_entries + res['data_entries']
                self.lumi_list += res['lumi_list']

                all_filenames = np.array(self.filesets[sample][sample]['files'])
                all_filenames_chunked = np.array_split(all_filenames, nchunks)
                for i in range(nchunks):
                    if len(all_filenames_chunked[i])>0:
                        files_i = {'files': all_filenames_chunked[i], 'treename': 'Events'}
                        self.filesets_chunked[sample].append({sample: files_i})

        if self.data_entries:
            print()
            data_entries_total = self.lumi_data[self.year]['events']
            print(f"Total events in {self.year}: {data_entries_total}")

            print(f"Loaded {self.data_entries} of {self.year} data events")
            prc = round(self.data_entries/data_entries_total*100, 2)
            print(f"This is ~ {prc}% of {self.year} data.")
            lumi_data = LumiData(f"data/lumimasks/lumi{self.year}.csv")
 #           print(self.lumi_list.array)

#            self.lumi = lumi_data.get_lumi(self.lumi_list)
            print(f"Integrated luminosity: {self.lumi}/pb")
            print()
        if self.missing_samples:
            print(f"Missing samples: {self.missing_samples}")

        t1 = time.time()
        dt=round(t1-t0, 2)
        print(f"Loading took {dt} s")

        self.data_samples = [s for s in self.samples if 'data' in s]
        self.mc_samples = [s for s in self.samples if not ('data' in s)]
        self.datasets_to_save_unbin += self.data_samples


    def load_sample(self, sample, parallelize=1):
        import glob, tqdm
        import uproot
        import multiprocessing as mp
        print("Loading", sample)

        if sample not in self.paths:
            print(f"Couldn't load {sample}! Skipping.")
            return {'sample': sample, 'metadata': {}, 'files': {}, 'data_entries': 0, 'is_missing': True}

        all_files = []
        metadata = {}
        data_entries = 0
        data_runs = []
        data_lumis = []
        lumi_list = LumiList()
        
        if self.at_purdue:
            all_files = read_via_xrootd(self.server, self.paths[sample])
        else:
            all_files = [server+f for f in glob.glob(self.paths[sample]+'/**/**/*.root')]

#        if 'ttjets_sl' in sample:
#            all_files = all_files[0:10]
            
        if self.debug:
            all_files = [all_files[0]]
#            all_files = [all_files[31]]

        sumGenWgts = 0
        nGenEvts = 0
            
        if parallelize>1:
            pool = mp.Pool(parallelize)
            if 'data' in sample:
                a = [pool.apply_async(self.get_data, args=(f,)) for f in all_files]
            else:
                a = [pool.apply_async(self.get_mc, args=(f,)) for f in all_files]
            results = []
            for process in a:
                process.wait()
                results.append(process.get())
                pool.close()
            for ret in results:
                if 'data' in sample:
                    data_entries += ret['data_entries']
                    lumi_list += ret['lumi_list']
                else:
                    sumGenWgts += ret['sumGenWgts']
                    nGenEvts += ret['nGenEvts']
        else:
            for f in all_files:
                if 'data' in sample:
                    tree = uproot.open(f)['Events']
                    data_entries += tree.numentries
                    lumi_mask = LumiMask(self.parameters['lumimask'])
                    lumi_filter = lumi_mask(tree.array('run'), tree.array('luminosityBlock'))
                    lumi_list += LumiList(tree.array('run')[lumi_filter], tree.array('luminosityBlock')[lumi_filter])
                else:
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
        return {'sample': sample, 'metadata': metadata, 'files': files,\
                'data_entries':data_entries, 'lumi_list':lumi_list, 'is_missing':False}

    def get_data(self, f):
        import uproot
        from coffea.lumi_tools import LumiData, LumiList
        import numpy as np
        ret = {}
        file = uproot.open(f)
        tree = file['Events']
        ret['data_entries'] = tree.numentries
        
        bl = file.get("LuminosityBlocks")
        runs = bl.array("run")
        lumis = bl.array("luminosityBlock")
        lumi_mask = LumiMask(self.parameters['lumimask'])
        lumi_filter = lumi_mask(runs, lumis)
        #print("Lumi filter eff.: ",lumi_filter.mean())
        if len(runs[lumi_filter])>0:
            ret['lumi_list'] = LumiList(runs[lumi_filter], lumis[lumi_filter])
        else:
            ret['lumi_list'] = LumiList()
        return ret

    def get_mc(self, f):
        import uproot
        from coffea.lumi_tools import LumiData, LumiList
        ret = {}
        tree = uproot.open(f)['Runs']
        if 'NanoAODv6' in f:
            ret['sumGenWgts'] = tree.array('genEventSumw_')[0]
            ret['nGenEvts'] = tree.array('genEventCount_')[0]
        else:
            ret['sumGenWgts'] = tree.array('genEventSumw')[0]
            ret['nGenEvts'] = tree.array('genEventCount')[0]
        return ret

    
    def compute_lumi_weights(self):
        from config.cross_sections import cross_sections
        import json
        self.lumi_weights = {'data':1}
        for sample in self.mc_samples:
            N = self.metadata[sample]['sumGenWgts']
            if isinstance(cross_sections[sample], dict):
                xsec = cross_sections[sample][self.year]
            else:
                xsec = cross_sections[sample]
            if N>0:
                self.lumi_weights[sample] = xsec*self.lumi / N
            else:
                self.lumi_weights[sample] = 0
            lumi_wgt = self.lumi_weights[sample]
            print(f"{sample}: xsec={xsec}, N={N}, lumi_wgt={lumi_wgt}")
