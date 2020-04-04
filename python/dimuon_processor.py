from coffea import hist, util
from coffea.analysis_objects import JaggedCandidateArray, JaggedCandidateMethods
import coffea.processor as processor
from coffea.lookup_tools import extractor, dense_lookup, txt_converters, rochester_lookup
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty, JetTransformer, JetResolution, JetResolutionScaleFactor
from coffea.btag_tools import BTagScaleFactor
from coffea.lumi_tools import LumiMask

import awkward
from awkward import JaggedArray
import uproot
import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)
import numba
import pandas as pd

from python.utils import apply_roccor, p4_sum, p4_sum_alt, NNLOPS_Evaluator
from python.timer import Timer
from python.samples_info import SamplesInfo

import gc

def get_regions(mass):
    regions = {
        "z-peak": ((mass>76) & (mass<106)),
        "h-sidebands": ((mass>110) & (mass<115)) | ((mass>135) & (mass<150)),
        "h-peak": ((mass>115) & (mass<135)),
    }
    return regions

def add_systematic(weights_df, sys_name, w_nom, w_up, w_down):
    sys_name_up = f'{sys_name}_up'
    sys_name_down = f'{sys_name}_down'
    weights_df[sys_name_up] = weights_df['nominal']*w_up
    weights_df[sys_name_down] = weights_df['nominal']*w_down   
    if w_nom!=[]:
        weights_df.loc[:,(weights_df.columns!=sys_name_up)&(weights_df.columns!=sys_name_down)] =\
        weights_df.loc[:,(weights_df.columns!=sys_name_up)&(weights_df.columns!=sys_name_down)].multiply(w_nom, axis=0)

def get_jec_unc(name, jet_pt, jet_eta, jecunc):
    idx_func = jecunc.levels.index(name)
    jec_unc_func = jecunc._funcs[idx_func]
    function_signature = jecunc._funcs[idx_func].signature
    counts = jet_pt.counts
    args = {
        "JetPt": np.array(jet_pt.flatten()),
        "JetEta": np.array(jet_eta.flatten())
    }
    func_args = tuple([args[s] for s in function_signature])
    jec_unc_vec = jec_unc_func(*func_args)
    return awkward.JaggedArray.fromcounts(counts, jec_unc_vec)

def delta_r(eta1, eta2, phi1, phi2):
    deta = abs(eta1 - eta2)
    dphi = abs(np.mod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi)
    dr = np.sqrt(deta**2 + dphi**2)
    return deta, dphi, dr

def apply_geofit(muons_pt, muons_eta, muons_dxybs, muons_charge, year, mask):
    pt_cor = np.zeros(len(muons_pt.flatten()), dtype=float)
    d0_BS_charge_full = np.multiply(muons_dxybs.flatten(),muons_charge.flatten()) 
    passes_mask = (~mask) & (np.abs(d0_BS_charge_full)<999999.)
    d0_BS_charge = d0_BS_charge_full[passes_mask]
    pt = muons_pt.flatten()[passes_mask]
    eta = muons_eta.flatten()[passes_mask]
    
    pt_cor_mask = pt_cor[passes_mask]
    
    cuts = {
        'eta_1': (np.abs(eta) < 0.9),
        'eta_2': ((np.abs(eta) < 1.7) & (np.abs(eta) >= 0.9)),
        'eta_3': (np.abs(eta) >= 1.7)
    }
    
    factors = {
        '2016': {
            'eta_1': 411.34,
            'eta_2': 673.40,
            'eta_3': 1099.0,            
        },
        '2017': {
            'eta_1': 582.32,
            'eta_2': 974.05,
            'eta_3': 1263.4,            
        },        
        '2018': {
            'eta_1': 650.84,
            'eta_2': 988.37,
            'eta_3': 1484.6,            
        }
    }
    
    for eta_i in ['eta_1', 'eta_2', 'eta_3']:
        pt_cor_mask[cuts[eta_i]] = factors[year][eta_i]*d0_BS_charge[cuts[eta_i]]*pt[cuts[eta_i]]**2/10000.0
    pt_cor[passes_mask] = pt_cor_mask
    return (muons_pt.flatten() - pt_cor)
    
# https://github.com/jpata/hepaccelerate-cms/blob/f5965648f8a7861cb9856d0b5dd34a53ed42c027/tests/hmm/hmumu_utils.py#L1396
@numba.njit(parallel=True)
def correct_muon_with_fsr(muons_offsets, fsr_offsets, muons_pt, muons_pt_raw, muons_eta, muons_phi,\
                          muons_mass, muons_iso, muons_fsrIndex,
                            fsr_pt, fsr_eta, fsr_phi):    
    for iev in numba.prange(len(muons_offsets) - 1):
        #loop over muons in event
        mu_first = muons_offsets[iev]
        mu_last = muons_offsets[iev + 1]
        for imu in range(mu_first, mu_last):
            #relative FSR index in the event
            fsr_idx_relative = muons_fsrIndex[imu]

            if (fsr_idx_relative >= 0) and (muons_pt_raw[imu]>20):
                #absolute index in the full FSR vector for all events
                ifsr = fsr_offsets[iev] + fsr_idx_relative
                mu_kin = {"pt": muons_pt[imu], "eta": muons_eta[imu], "phi": muons_phi[imu], "mass": muons_mass[imu]}
                fsr_kin = {"pt": fsr_pt[ifsr], "eta": fsr_eta[ifsr], "phi": fsr_phi[ifsr],"mass": 0.}

                # dR between muon and photon
                deta = muons_eta[imu] - fsr_eta[ifsr]
                dphi = np.mod(muons_phi[imu] - fsr_phi[ifsr] + np.pi, 2*np.pi) - np.pi
                dr = np.sqrt(deta**2 + dphi**2)

                update_iso = dr<0.4

                #reference: https://gitlab.cern.ch/uhh-cmssw/fsr-photon-recovery/tree/master
                if update_iso:
                    muons_iso[imu] = (muons_iso[imu]*muons_pt[imu] - fsr_pt[ifsr])/muons_pt[imu]
                    
                #compute and set corrected momentum
                px_total = 0
                py_total = 0
                pz_total = 0
                e_total = 0
                for obj in [mu_kin, fsr_kin]:
                    px = obj["pt"] * np.cos(obj["phi"])
                    py = obj["pt"] * np.sin(obj["phi"])
                    pz = obj["pt"] * np.sinh(obj["eta"])
                    e = np.sqrt(px**2 + py**2 + pz**2 + obj["mass"]**2)
                    px_total += px
                    py_total += py
                    pz_total += pz
                    e_total += e
                out_pt = np.sqrt(px_total**2 + py_total**2)
                out_eta = np.arcsinh(pz_total / out_pt)
                out_phi = np.arctan2(py_total, px_total)

                muons_pt[imu] = out_pt
                muons_eta[imu] = out_eta
                muons_phi[imu] = out_phi

    return  muons_pt, muons_eta, muons_phi, muons_mass, muons_iso
    
# Look at ProcessorABC documentation to see the expected methods and what they are supposed to do
# https://coffeateam.github.io/coffea/api/coffea.processor.ProcessorABC.html
class DimuonProcessor(processor.ProcessorABC):
    def __init__(self, samp_info, do_roch=True, do_fsr=True, evaluate_dnn=False,\
                 do_timer=False, save_unbin=True, do_lheweights=True, do_geofit=False, do_jec=True,\
                 do_jer=True, do_jecunc=False, do_nnlops=True, do_btagsf=False, do_pdf=True, debug=False): 
        from config.parameters import parameters
        from config.variables import variables
        if not samp_info:
            print("Samples info missing!")
            return
        self.samp_info = samp_info
        self.year = self.samp_info.year

#        self.mass_window = mass_window
        self.mass_window = [76, 150]
        self.do_roch = do_roch
        self.save_unbin = save_unbin
        self.do_geofit = do_geofit
        self.do_jec = do_jec
        self.do_jer = do_jer
        self.do_jecunc = do_jecunc
        self.do_pdf = do_pdf
        self.debug = debug
        
        self.do_btagsf = do_btagsf
        self.do_nnlops = do_nnlops
        self.do_fsr = do_fsr
        self.evaluate_dnn = evaluate_dnn
        self.do_lheweights = do_lheweights
        self.parameters = {k:v[self.year] for k,v in parameters.items()}
        self.timer = Timer('global') if do_timer else None
        
        self._columns = self.parameters["proc_columns"]
        self.variations = ['nominal', 'pu_weight', 'muSF', 'l1prefiring_weight']
        
        dataset_axis = hist.Cat("dataset", "")
        region_axis = hist.Cat("region", "") # Z-peak, Higgs SB, Higgs peak
        channel_axis = hist.Cat("channel", "") # ggh or VBF  
        syst_axis = hist.Cat("syst", "")
                        
        self.regions = self.samp_info.regions
        self.channels = self.samp_info.channels

        self.overlapping_samples = self.samp_info.overlapping_samples
        self.specific_samples = self.samp_info.specific_samples
        self.datasets_to_save_unbin = self.samp_info.datasets_to_save_unbin
        self.lumi_weights = self.samp_info.lumi_weights
        
        from config.variables import Variable
        for syst in self.variations:
            if 'nominal' in syst:
                variables.append(Variable("weight_nominal", "weight_nominal", 1, 0, 1))
            else:
                variables.append(Variable(f"weight_{syst}_up", f"weight_{syst}_up", 1, 0, 1))
                variables.append(Variable(f"weight_{syst}_down", f"weight_{syst}_down", 1, 0, 1))    

        if self.evaluate_dnn:
            variables.append(Variable(f"dnn_score_nominal", f"dnn_score_nominal", 12, 0, self.parameters["dnn_max"]))
            if self.do_jecunc:
                for v_name in self.parameters["jec_unc_to_consider"]:
                    variables.append(Variable(f"dnn_score_{v_name}_up", f"dnn_score_{v_name}_up", 12, 0, self.parameters["dnn_max"]))
                    variables.append(Variable(f"dnn_score_{v_name}_down", f"dnn_score_{v_name}_down", 12, 0, self.parameters["dnn_max"]))
                
        for i in range(9):
            variables.append(Variable(f"LHEScaleWeight_{i}", f"LHEScaleWeight_{i}", 1, 0, 1))
        
        if self.do_pdf:
            variables.append(Variable("pdf_rms", "pdf_rms", 1, 0, 1))
        
        self.vars_unbin = [v.name for v in variables]
        ### Prepare accumulators for binned output ###
        bin_dict = {}
        for v in variables:
            if v.name=='dimuon_mass':
                axis = hist.Bin(v.name, v.caption, v.nbins, self.mass_window[0], self.mass_window[1])
            else:
                axis = hist.Bin(v.name, v.caption, v.nbins, v.xmin, v.xmax)
            bin_dict[v.name] = hist.Hist("Counts", dataset_axis, region_axis, channel_axis, syst_axis, axis)  
        accumulator_binned = processor.dict_accumulator(bin_dict)
        ### --------------------------------------- ###
        
        ### Prepare accumulators for unbinned output ###
        unbin_dict = {}
        if self.save_unbin:
            for p in self.samp_info.samples:
                for v in self.vars_unbin:
                    if 'dnn_score' in v: continue
                    for c in self.channels:
                        for r in self.regions:
#                            print(v,p,c,r)
                            unbin_dict[f'{v}_unbin_{p}_c_{c}_r_{r}'] = processor.column_accumulator(np.ndarray([]))
                            # have to encode everything into the name because having multiple axes isn't possible
        accumulator_unbinned = processor.dict_accumulator(unbin_dict)
        ### --------------------------------------- ###
        
        event_weights = {}
        for c in self.channels:
            for r in self.regions:
                event_weights[f'event_weight_{c}_{r}'] = processor.column_accumulator(np.ndarray([]))
        
        acc_dicts = {'binned':accumulator_binned, 'unbinned':accumulator_unbinned}
        acc_dicts.update(**event_weights)
        
        accumulators = processor.dict_accumulator(acc_dicts)
        self._accumulator = accumulators
        
        mu_id_vals = 0
        mu_id_err = 0
        mu_iso_vals = 0
        mu_iso_err = 0
        mu_trig_vals_data = 0
        mu_trig_err_data = 0
        mu_trig_vals_mc = 0
        mu_trig_err_mc = 0

        for scaleFactors in self.parameters['muSFFileList']:
            id_file = uproot.open(scaleFactors['id'][0])
            iso_file = uproot.open(scaleFactors['iso'][0])
            trig_file = uproot.open(scaleFactors['trig'][0])
            
            mu_id_vals += id_file[scaleFactors['id'][1]].values * scaleFactors['scale']
            mu_id_err += id_file[scaleFactors['id'][1]].variances**0.5 * scaleFactors['scale']
            mu_id_edges = id_file[scaleFactors['id'][1]].edges

            mu_iso_vals += iso_file[scaleFactors['iso'][1]].values * scaleFactors['scale']
            mu_iso_err += iso_file[scaleFactors['iso'][1]].variances**0.5 * scaleFactors['scale']
            mu_iso_edges = iso_file[scaleFactors['iso'][1]].edges

            mu_trig_vals_data += trig_file[scaleFactors['trig'][1]].values * scaleFactors['scale']
            mu_trig_vals_mc += trig_file[scaleFactors['trig'][2]].values * scaleFactors['scale']
            mu_trig_err_data += trig_file[scaleFactors['trig'][1]].variances**0.5 * scaleFactors['scale']
            mu_trig_err_mc += trig_file[scaleFactors['trig'][2]].variances**0.5 * scaleFactors['scale']
            mu_trig_edges = trig_file[scaleFactors['trig'][1]].edges

        self.mu_id_sf = dense_lookup.dense_lookup(mu_id_vals, mu_id_edges)
        self.mu_id_err = dense_lookup.dense_lookup(mu_id_err, mu_id_edges)
        self.mu_iso_sf = dense_lookup.dense_lookup(mu_iso_vals, mu_iso_edges)
        self.mu_iso_err = dense_lookup.dense_lookup(mu_iso_err, mu_iso_edges)

        self.mu_trig_eff_data = dense_lookup.dense_lookup(mu_trig_vals_data, mu_trig_edges)
        self.mu_trig_eff_mc = dense_lookup.dense_lookup(mu_trig_vals_mc, mu_trig_edges)
        
        self.mu_trig_err_data = dense_lookup.dense_lookup(mu_trig_err_data, mu_trig_edges)    
        self.mu_trig_err_mc = dense_lookup.dense_lookup(mu_trig_err_mc, mu_trig_edges)    
        
        zpt_filename = self.parameters['zpt_weights_file']
        puid_filename = self.parameters['puid_sf_file']
        rescalib_files = []
        
        self.extractor = extractor()
        self.extractor.add_weight_sets([f"* * {zpt_filename}"])
        self.extractor.add_weight_sets([f"* * {puid_filename}"])
        
        for mode in ["Data", "MC"]:
            label = f"res_calib_{mode}_{self.year}"
            path = self.parameters['res_calib_path']       
            file_path = f"{path}/{label}.root"
            self.extractor.add_weight_sets([f"{label} {label} {file_path}"])
            
        self.extractor.finalize()
        self.evaluator = self.extractor.make_evaluator()
        
        if '2016' in self.year:
            self.zpt_path = 'zpt_weights/2016_value'
        else:
            self.zpt_path = 'zpt_weights/2017_value'
        self.evaluator[self.zpt_path]._axes = self.evaluator[self.zpt_path]._axes[0]            
#https://github.com/CoffeaTeam/coffea/blob/2650ad7657094f6e50ebf962a1fc1763cd2c6601/coffea/lookup_tools/dense_lookup.py#L37
    
        rochester_data = txt_converters.convert_rochester_file(self.parameters["roccor_file"], loaduncs=True)
        self.rochester = rochester_lookup.rochester_lookup(rochester_data)
        
        self.puLookup = util.load(self.parameters['puLookup'])
        self.puLookup_Up = util.load(self.parameters['puLookup_Up'])
        self.puLookup_Down = util.load(self.parameters['puLookup_Down'])
        
        jetext = extractor()
        jetext.add_weight_sets(self.parameters['jec_weight_sets'])
        jetext.finalize()
        Jetevaluator = jetext.make_evaluator()
        JECcorrector = FactorizedJetCorrector(**{name: Jetevaluator[name] for name in self.parameters['jec_names']})
        JECuncertainties = JetCorrectionUncertainty(**{name:Jetevaluator[name] for name in self.parameters['junc_names']})
        JER = JetResolution(**{name:Jetevaluator[name] for name in self.parameters['jer_names']})
        JERsf = JetResolutionScaleFactor(**{name:Jetevaluator[name] for name in self.parameters['jersf_names']})
        if self.do_jer:
            self.Jet_transformer = JetTransformer(jec=JECcorrector,junc=JECuncertainties, jer = JER, jersf = JERsf)
        else:
            self.Jet_transformer = JetTransformer(jec=JECcorrector,junc=JECuncertainties)
        self.Jet_transformer_data = {}
        self.data_runs = list(self.parameters['jec_unc_names_data'].keys())
        for run in self.data_runs:
            JECcorrector_Data = FactorizedJetCorrector(**{name: Jetevaluator[name] for name in\
                                                          self.parameters['jec_names_data'][run]})
            JECuncertainties_Data = JetCorrectionUncertainty(**{name:Jetevaluator[name] for name in\
                                                                self.parameters['junc_names_data'][run]})

            self.Jet_transformer_data[run] = JetTransformer(jec=JECcorrector_Data,junc=JECuncertainties_Data)
            
        all_jec_names = [name for name in dir(Jetevaluator) if self.parameters['jec_unc_sources'] in name]
        self.JECuncertaintySources = JetCorrectionUncertainty(**{name: Jetevaluator[name] for name in all_jec_names})
        self.jet_unc_names = list(self.JECuncertaintySources.levels)
        if self.do_btagsf:
            self.btag_sf = BTagScaleFactor(self.parameters["btag_sf_csv"], BTagScaleFactor.RESHAPE,\
                                           'iterativefit,iterativefit,iterativefit')
            
#            btag_files = {
#                "2016": "DeepCSV_102XSF_V1.btag.csv",
#                "2017": "DeepCSV_2016LegacySF_V1.btag.csv",
#                "2018": "DeepCSV_94XSF_V5_B_F.btag.csv",
#            }
#            btag_file = btag_files[self.year]
#            btag_ext = extractor()
#            btag_ext.add_weight_sets([f"btag{self.year} * data/btag/{btag_file}"])
#            btag_ext.finalize()
#            self.btag_csv = btag_ext.make_evaluator()
#            import pickle
#            with open(f'data/btag/eff_lookup_{self.year}_L.pkl', 'rb') as _file:
#                self.eff_lookup_L = pickle.load(_file)
#            with open(f'data/btag/eff_lookup_{self.year}_M.pkl', 'rb') as _file:
#                self.eff_lookup_M = pickle.load(_file)

                
    @property
    def accumulator(self):
        return self._accumulator
    
    @property
    def columns(self):
        return self._columns
    
    def process(self, df):
        # TODO: Add systematic uncertainties
        
        # Variables to add (all need to be saved for unbinned analysis):
        # dimuon_phiCS
        
        #---------------------------------------------------------------#        
        # Filter out events not passing HLT or having less than 2 muons.
        #---------------------------------------------------------------#
#        self.test_event = 213589
        self.test_event = 223846166
#        self.debug=False
#        self.timer=None
        for ev in [self.test_event]:
            if self.test_event in df.event:
                print(ev)
                self.debug = True
                self.timer = Timer('global')
        if self.timer:
            self.timer.update() 
            
        output = self.accumulator.identity()
        dataset = df.metadata['dataset']
        is_mc = 'data' not in dataset
        
        if self.debug:
            print("Input: ", df.shape[0])
                
        hlt = np.zeros(df.shape[0], dtype=bool)
        for hlt_path in self.parameters['hlt']:
            if hlt_path in df.HLT.columns:
                hlt = hlt | df.HLT[hlt_path]
        
        df = df[hlt&(df.Muon.counts>1)]
    
        if self.debug:
            print("Passing HLT and at least 2 muons (any): ", len(df))
            
            
        #---------------------------------------------------------------#  
        # From now on, number of events will remain unchanged (size of 'mask').
        # Variable 'mask' will be used to store global event selection        
        # Apply HLT, lumimask, genweights, PU weights and L1 prefiring weights
        #---------------------------------------------------------------#            
        
        numevents = df.shape[0]
        weights = pd.DataFrame(1, index=np.arange(numevents), columns=['nominal'])
        
        if is_mc:    
            mask = np.ones(numevents, dtype=bool)
            genweight = df.genWeight.flatten()
            weights = weights.multiply(genweight, axis=0)

            if '2016' in self.year:
                pu_weight = self.puLookup(dataset, df.Pileup.nTrueInt)
                pu_weight_up = self.puLookup_Up(dataset, df.Pileup.nTrueInt)
                pu_weight_down = self.puLookup_Down(dataset, df.Pileup.nTrueInt)
            else:
                pu_weight = self.puLookup(df.Pileup.nTrueInt)
                pu_weight_up = self.puLookup_Up(df.Pileup.nTrueInt)
                pu_weight_down = self.puLookup_Down(df.Pileup.nTrueInt)
                
            add_systematic(weights, 'pu_weight', pu_weight, pu_weight_up, pu_weight_down)

            if dataset in self.lumi_weights:
                weights = weights.multiply(self.lumi_weights[dataset], axis=0)

            if self.parameters["do_l1prefiring_wgts"]:
                prefiring_wgt = df.L1PreFiringWeight.Nom.flatten()
                add_systematic(weights, 'l1prefiring_weight', prefiring_wgt,\
                               df.L1PreFiringWeight.Up.flatten(), df.L1PreFiringWeight.Dn.flatten()) 
                if self.debug:
                    print('Prefiring weight: ', prefiring_wgt[df.event==self.test_event])
                    print('Avg. prefiring weight: ', prefiring_wgt.mean())

            if self.debug:
                print('Gen weight: ', genweight[df.event==self.test_event])
                print('Avg. Gen weight: ', genweight.mean())
                print('PU weight: ', pu_weight[df.event==self.test_event])
                print('Avg. pu weight: ', pu_weight.mean())
                
        else:
            lumi_info = LumiMask(self.parameters['lumimask'])
            mask = lumi_info(df.run, df.luminosityBlock)
            if self.debug:
                print("Pass lumimask", sum(mask))
                      
        if self.timer:
            self.timer.add_checkpoint("Applied HLT and lumimask")
            

        #---------------------------------------------------------------#        
        # Update muon kinematics with Rochester correction, FSR recovery and GeoFit correction
        # Raw pT and eta are stored to be used in event selection
        #---------------------------------------------------------------# 
            
        mu = df.Muon
        muons_pt = mu.pt.flatten()
        muons_pt_raw = muons_pt
        muons_eta = mu.eta.flatten()
        muons_eta_raw = muons_eta
        muons_phi = mu.phi.flatten()
        muons_mass = mu.mass.flatten()
        muons_iso = mu.pfRelIso04_all.flatten()          
        has_fsr_photon_mask = np.zeros(len(mu.pt.flatten()), dtype=bool)
        
        if self.do_roch:
            roch_corr = apply_roccor(self.rochester, is_mc, mu).flatten()
        else:
            roch_corr = np.ones(len(muons_pt), dtype=float)

        muons_pt = muons_pt*roch_corr
        muons_pt_raw = muons_pt # Rochester should be still applied

        if self.do_fsr:
            fsr = df.FsrPhoton
            fsr_offsets = fsr.counts2offsets(fsr.counts)
            muons_offsets = mu.counts2offsets(mu.counts)
            fsr_pt = np.array(fsr.pt.flatten(), dtype=float)
            fsr_eta = np.array(fsr.eta.flatten(), dtype=float)
            fsr_phi = np.array(fsr.phi.flatten(), dtype=float)
            muons_fsrIndex = np.array(mu.fsrPhotonIdx.flatten(), dtype=int)

            has_fsr_photon_mask = muons_fsrIndex>=0
            muons_pt, muons_eta, muons_phi, muons_mass, muons_iso = correct_muon_with_fsr(muons_offsets, fsr_offsets,\
                                  np.array(muons_pt, dtype=float), np.array(muons_pt_raw, dtype=float),\
                                  np.array(muons_eta, dtype=float), np.array(muons_phi, dtype=float),\
                                  np.array(muons_mass, dtype=float), np.array(muons_iso, dtype=float),\
                                  np.array(muons_fsrIndex, dtype=int), fsr_pt, fsr_eta, fsr_phi) 

        # GeoFit correction
        if self.do_geofit:
            muons_dxybs = mu.dxybs.flatten()
            muons_charge = mu.charge.flatten()
            muons_pt = apply_geofit(muons_pt, muons_eta, muons_dxybs, muons_charge, self.year, has_fsr_photon_mask).flatten() 
            
        updated_attrs = {'pt': muons_pt, 'pt_raw': muons_pt_raw, 'eta':muons_eta, 'eta_raw':muons_eta_raw,\
                         'phi':muons_phi, 'mass':muons_mass, 'pfRelIso04_all':muons_iso}
        
        muonarrays = {key:mu[key].flatten() for key in mu.columns}
        muonarrays.update(updated_attrs)
        muons = JaggedCandidateArray.candidatesfromcounts(mu.counts, **muonarrays)
        updated_attrs.clear()
        
        #---------------------------------------------------------------#        
        # Select muons that pass pT, eta, isolation cuts, muon ID and quality flags
        # Select events with 2 OS muons, no electrons, passing quality cuts and at least one good PV
        #---------------------------------------------------------------# 
        
        pass_event_flags = np.ones(numevents, dtype=bool)
        for flag in self.parameters["event_flags"]:
            pass_event_flags = pass_event_flags & df.Flag[flag]
        if self.debug:
            print("Pass event flags:", sum(pass_event_flags))

        pass_muon_flags = np.ones(df.shape[0], dtype=bool)
        for flag in self.parameters["muon_flags"]:
            pass_muon_flags = pass_muon_flags & muons[flag]
            
        muons = muons[(muons.pt_raw > self.parameters["muon_pt_cut"]) &\
                      (abs(muons.eta_raw) < self.parameters["muon_eta_cut"]) &\
                        (muons.pfRelIso04_all < self.parameters["muon_iso_cut"]) &\
                        muons[self.parameters["muon_id"]] 
                     & pass_muon_flags]    
        
        two_os_muons = ((muons.counts == 2) & (muons['charge'].prod() == -1))
        
        electrons = df.Electron[(df.Electron.pt > self.parameters["electron_pt_cut"]) &\
                                     (abs(df.Electron.eta) < self.parameters["electron_eta_cut"]) &\
                                     (df.Electron[self.parameters["electron_id"]] == 1)]
                
        electron_veto = (electrons.counts>-1)
        good_pv = (df.PV.npvsGood > 0)
        
        event_filter = (pass_event_flags & two_os_muons & electron_veto & good_pv).flatten()
        
        if self.debug:
            print("Has 2 OS muons passing selections, good PV and no electrons:", sum(event_filter))

        mask = mask & event_filter

        if self.timer:
            self.timer.add_checkpoint("Applied preselection")

        #---------------------------------------------------------------#        
        # Initialize muon variables
        #---------------------------------------------------------------# 
        
        mu1 = muons[muons.pt.argmax()]
        mu2 = muons[muons.pt.argmin()]
       
        mu1_variable_names = ['mu1_pt', 'mu1_pt_over_mass', 'mu1_eta', 'mu1_phi', 'mu1_iso']
        mu1_variables = {}
        for n in mu1_variable_names:
            mu1_variables[n] = np.zeros(numevents)
            
        mu2_variable_names = ['mu2_pt', 'mu2_pt_over_mass', 'mu2_eta', 'mu2_phi', 'mu2_iso']
        mu2_variables = {}
        for n in mu2_variable_names:
            mu2_variables[n] = np.zeros(numevents)
        
        #---------------------------------------------------------------#        
        # Select events with muons passing leading pT cut and trigger matching
        #---------------------------------------------------------------# 

        # Events where there is at least one muon passing leading muon pT cut (redundant selection)
        pass_leading_pt = np.zeros(numevents, dtype=bool)
        pass_leading_pt[muons.counts>0] = (mu1.pt_raw>self.parameters["muon_leading_pt"]).flatten()
        
        # All L3 trigger muons
        trigmuarrays = {key:df.TrigObj[key].flatten() for key in df.TrigObj.columns}
        trigmuarrays.update({'mass':0})
        trig_muons = JaggedCandidateArray.candidatesfromcounts(df.TrigObj.counts, **trigmuarrays)
        trig_muons = trig_muons[trig_muons.id == 13]
        trigmuarrays.clear()
        
        # Muons that pass tight id and iso as well as leading muon pT cut     
        mu_for_trigmatch = muons[(muons.pt_raw > self.parameters["muon_leading_pt"]) &\
                                   (muons.pfRelIso04_all < self.parameters["muon_trigmatch_iso"]) &\
                                   muons[self.parameters["muon_trigmatch_id"]]]
        
        # For every such muon check if there is a L3 object within dR<0.1
        muTrig = mu_for_trigmatch.cross(trig_muons, nested = True)
        _,_,dr = delta_r(muTrig.i0.eta_raw, muTrig.i1.eta, muTrig.i0.phi, muTrig.i1.phi)
        has_matched_trigmuon = (dr < self.parameters["muon_trigmatch_dr"]).any()
        
        # Events where there is a trigger object matched to a tight-ID tight-Iso muon passing leading pT cut
        event_passing_trig_match = (mu_for_trigmatch[has_matched_trigmuon].counts>0).flatten()
        
        mask = mask & pass_leading_pt & event_passing_trig_match
        
        if self.debug:
            print("Leading pT cut, trigger matching:", sum(mask))
        
        #---------------------------------------------------------------#        
        # Initialize and fill dimuon and muon variables
        #---------------------------------------------------------------# 
        
        two_muons = muons.counts==2
        
        dimuon_variable_names = ['dimuon_mass', 'dimuon_mass_res', 'dimuon_mass_res_rel', 'dimuon_pt', 'dimuon_eta',\
                                 'dimuon_phi', 'dimuon_dEta', 'dimuon_dPhi', 'dimuon_dR', 'dimuon_cosThetaCS']
        dimuon_variables = {}
        for n in dimuon_variable_names:
            dimuon_variables[n] = np.zeros(numevents)

        dimuon_variables['dimuon_pt'][two_muons],\
        dimuon_variables['dimuon_eta'][two_muons],\
        dimuon_variables['dimuon_phi'][two_muons],\
        dimuon_variables['dimuon_mass'][two_muons] = p4_sum(mu1[two_muons], mu2[two_muons])

        dimuon_variables['dimuon_dEta'][two_muons],\
        dimuon_variables['dimuon_dPhi'][two_muons],\
        dimuon_variables['dimuon_dR'][two_muons] = delta_r(mu1[two_muons].eta.flatten(),\
                                                                                 mu2[two_muons].eta.flatten(),\
                                                                                 mu1[two_muons].phi.flatten(),\
                                                                                 mu2[two_muons].phi.flatten())

        dpt1 = (mu1[two_muons].ptErr*dimuon_variables['dimuon_mass'][two_muons]) / (2*mu1[two_muons].pt)
        dpt2 = (mu2[two_muons].ptErr*dimuon_variables['dimuon_mass'][two_muons]) / (2*mu2[two_muons].pt)
        
        if is_mc:
            label = f"res_calib_MC_{self.year}"
        else:
            label = f"res_calib_Data_{self.year}"
            
        calibration = np.array(self.evaluator[label](mu1[two_muons].pt.flatten(),\
                                                     abs(mu1[two_muons].eta.flatten()),\
                                                     abs(mu2[two_muons].eta.flatten())))
        
        dimuon_variables['dimuon_mass_res'][two_muons] = (np.sqrt(dpt1 * dpt1 + dpt2 * dpt2) * calibration).flatten()
        dimuon_variables['dimuon_mass_res_rel'][two_muons] = (dimuon_variables['dimuon_mass_res'][two_muons] /\
                                                              dimuon_variables['dimuon_mass'][two_muons]).flatten()
        
            
        mu1_px = mu1.pt*np.cos(mu1.phi)
        mu1_py = mu1.pt*np.sin(mu1.phi)
        mu1_pz = mu1.pt*np.sinh(mu1.eta)
        mu1_e  = np.sqrt(mu1_px**2 + mu1_py**2 + mu1_pz**2 + mu1.mass**2)
        mu2_px = mu2.pt*np.cos(mu2.phi)
        mu2_py = mu2.pt*np.sin(mu2.phi)
        mu2_pz = mu2.pt*np.sinh(mu2.eta)
        mu2_e  = np.sqrt(mu2_px**2 + mu2_py**2 + mu2_pz**2 + mu2.mass**2)
        
        dimuon_variables['dimuon_cosThetaCS'][two_muons] =\
                2*( mu1_pz[two_muons]*mu2_e[two_muons] - mu1_e[two_muons]*mu2_pz[two_muons] ) / \
                  ( dimuon_variables['dimuon_mass'][two_muons]*\
                   np.sqrt(dimuon_variables['dimuon_mass'][two_muons]*dimuon_variables['dimuon_mass'][two_muons] +\
                           dimuon_variables['dimuon_pt'][two_muons]*dimuon_variables['dimuon_pt'][two_muons]) )

        mu1_variables['mu1_pt'][two_muons] = mu1[two_muons].pt.flatten()
        mu1_variables['mu1_eta'][two_muons] = mu1[two_muons].eta.flatten()
        mu1_variables['mu1_phi'][two_muons] = mu1[two_muons].phi.flatten()
        mu1_variables['mu1_iso'][two_muons] = mu1[two_muons].pfRelIso04_all.flatten()
        mu1_variables['mu1_pt_over_mass'][two_muons] = np.divide(mu1[two_muons].pt.flatten(),\
                                                                 dimuon_variables['dimuon_mass'][two_muons])
        
        mu2_variables['mu2_pt'][two_muons] = mu2[two_muons].pt.flatten()
        mu2_variables['mu2_eta'][two_muons] = mu2[two_muons].eta.flatten()
        mu2_variables['mu2_phi'][two_muons] = mu2[two_muons].phi.flatten()
        mu2_variables['mu2_iso'][two_muons] = mu2[two_muons].pfRelIso04_all.flatten()
        mu2_variables['mu2_pt_over_mass'][two_muons] = np.divide(mu2[two_muons].pt.flatten(),\
                                                                 dimuon_variables['dimuon_mass'][two_muons])
    
    
        muon_variables = dimuon_variables
        muon_variables.update(**mu1_variables)
        muon_variables.update(**mu2_variables)
        
        #---------------------------------------------------------------#        
        # Prepare jets
        #---------------------------------------------------------------# 
        
        jetarrays = {key:df.Jet[key].flatten() for key in df.Jet.columns} 
        jetarrays.update(**{'ptRaw':(df.Jet.pt * (1-df.Jet.rawFactor)).flatten(),\
                            'massRaw':(df.Jet.mass * (1-df.Jet.rawFactor)).flatten(),\
                            'rho': (df.Jet.pt.ones_like()*df.fixedGridRhoFastjetAll).flatten() })
        
        if is_mc:
            ptGenJet = df.Jet.pt.zeros_like()
            genJetIdx = df.Jet.genJetIdx
            try:
                gj_mask = genJetIdx<df.GenJet.counts
                ptGenJet[gj_mask] = df.GenJet[genJetIdx[gj_mask]].pt
            except:
                pass
            jetarrays.update(**{'ptGenJet': ptGenJet.flatten()})
            
        jets = JaggedCandidateArray.candidatesfromcounts(df.Jet.counts, **jetarrays)
        jetarrays.clear()
        
        if self.debug:
            print("Total jets: ", sum(jets.counts))
 
        #---------------------------------------------------------------#        
        # Apply JEC, get JEC variations
        #---------------------------------------------------------------#
    
        jet_variation_names = ['nominal']
        if self.do_jec: 
            cols = {'pt':'__fast_pt',
                    'eta':'__fast_eta',
                    'phi':'__fast_phi',
                    'mass':'__fast_mass'}
            cols.update(**{k:k for k in ['qgl','btagDeepB','ptRaw','massRaw','rho','area','jetId','puId']})
            if is_mc:
                cols.update({k:k for k in ['genJetIdx','ptGenJet','hadronFlavour']})
            jetarrays = {key:jets[v].flatten() for key, v in cols.items()}
            jets = JaggedCandidateArray.candidatesfromcounts(jets.counts, **jetarrays)
            jetarrays.clear()
            if self.do_jecunc:
                for junc_name in self.jet_unc_names:
                    if junc_name not in self.parameters["jec_unc_to_consider"]: continue
                    jet_variation_names += [f"{junc_name}_up", f"{junc_name}_down"]
            
                        
            if is_mc:
                self.Jet_transformer.transform(jets, forceStochastic=False)
                if self.do_jecunc:
                    for junc_name in self.jet_unc_names:
                        if junc_name not in self.parameters["jec_unc_to_consider"]: continue
                        jec_up_down = get_jec_unc(junc_name, jets.pt, jets.eta, self.JECuncertaintySources)
                        jec_corr_up, jec_corr_down = jec_up_down[:,:,0], jec_up_down[:,:,1]
                        pt_name_up = f"pt_{junc_name}_up"
                        pt_name_down = f"pt_{junc_name}_down"
                        jets.add_attributes(**{pt_name_up: jets.pt*jec_corr_up, pt_name_down: jets.pt*jec_corr_down})
            else:
                for run in self.data_runs: # 'B', 'C', 'D', etc...
                    if run in dataset: # dataset name is something like 'data_B'
                        self.Jet_transformer_data[run].transform(jets, forceStochastic=False)   

        jets.add_attributes(**{'pt_nominal': jets.pt})
        jets = jets[jets.pt.argsort()]  

        #---------------------------------------------------------------#        
        # Apply jetID
        #---------------------------------------------------------------#        

        # Jet ID
        if "loose" in self.parameters["jet_id"]:
            jet_id = (jets.jetId >= 1)
        elif "tight" in self.parameters["jet_id"]:
            if '2016' in self.year:
                jet_id = (jets.jetId >= 3)
            else:
                jet_id = (jets.jetId >= 2)
        else:
            jet_id = jets.ones_like()

        if self.debug:
            print("Jets passing ID: ", jet_id.flatten().sum())

            
        good_jet_id = jet_id & (jets.qgl > -2)
        
        jets = jets[good_jet_id]         
        
        
        #---------------------------------------------------------------#        
        # Calculate other event weights
        #---------------------------------------------------------------#        
        
        if self.do_nnlops and ('ggh' in dataset):
            nnlops = NNLOPS_Evaluator('data/NNLOPS_reweight.root')
            nnlopsw = np.ones(numevents, dtype=float)
            gen_njets = df.GenJet[df.GenJet.pt>30.].counts
            gen_higgs = df.GenPart[(df.GenPart.pdgId == 25)&(df.GenPart.status == 62)]
            has_higgs = gen_higgs.counts>0
            gen_hpt = np.zeros(numevents, dtype=float)
            gen_hpt[has_higgs] = gen_higgs[has_higgs].pt[:,0]
            if 'amc' in dataset:
                nnlopsw[has_higgs] = nnlops.evaluate(gen_hpt[has_higgs], gen_njets[has_higgs], "mcatnlo")
            elif 'powheg' in dataset:
                nnlopsw[has_higgs] = nnlops.evaluate(gen_hpt[has_higgs], gen_njets[has_higgs], "powheg")
            weights = weights.multiply(nnlopsw, axis=0)

        if 'dy' in dataset:
            zpt_weight = np.ones(numevents, dtype=float)
            zpt_weight[two_muons] = self.evaluator[self.zpt_path](dimuon_variables['dimuon_pt'][two_muons]).flatten()
            weights = weights.multiply(zpt_weight, axis=0)

            if self.debug:
                print("Avg. Zpt weight: ", zpt_weight[df.event==self.test_event])
                print("Avg. Zpt weight: ", zpt_weight.mean())
            
        if is_mc:
            pt = muons.pt_raw.compact()
            eta = muons.eta_raw.compact()
            abs_eta = abs(muons.eta_raw.compact())
            muID = np.ones(len(muons.flatten()), dtype=float)
            muIso = np.ones(len(muons.flatten()), dtype=float)
            muTrig = np.ones(numevents, dtype=float)
            muIDerr = np.zeros(len(muons.flatten()), dtype=float)
            muIsoerr = np.zeros(len(muons.flatten()), dtype=float)
            muTrig_up = np.ones(numevents, dtype=float)
            muTrig_down = np.ones(numevents, dtype=float)
            
            if '2016' in self.year:
                muID = self.mu_id_sf(eta, pt)
                muIso = self.mu_iso_sf(eta, pt)
                muIDerr = self.mu_id_err(eta, pt)
                muIsoerr = self.mu_iso_err(eta, pt) 
                muTrig_data = self.mu_trig_eff_data(abs_eta, pt)
                muTrig_mc = self.mu_trig_eff_mc(abs_eta, pt)
                muTrigerr_data = self.mu_trig_err_data(abs_eta, pt)
                muTrigerr_mc = self.mu_trig_err_mc(abs_eta, pt)
            else:
                muID = self.mu_id_sf(pt, abs_eta)
                muIso = self.mu_iso_sf(pt, abs_eta)
                muIDerr = self.mu_id_err(pt, abs_eta)
                muIsoerr = self.mu_iso_err(pt, abs_eta)
                muTrig_data = self.mu_trig_eff_data(abs_eta, pt)
                muTrig_mc = self.mu_trig_eff_mc(abs_eta, pt)                
                muTrigerr_data = self.mu_trig_err_data(abs_eta, pt)
                muTrigerr_mc = self.mu_trig_err_mc(abs_eta, pt)
            
            denom = ( (1 - (1. - muTrig_mc).prod()) )
            denom_up = ( (1 - (1. - muTrig_mc - muTrigerr_mc).prod()) != 0 )
            denom_dn = ( (1 - (1. - muTrig_mc + muTrigerr_mc).prod()) != 0 )
            
            muTrig[denom!=0] = ( (1 - (1. - muTrig_data).prod()) / denom )[denom!=0]
            muTrig_up[denom_up!=0] = ( (1 - (1. - muTrig_data - muTrigerr_data).prod()) / denom_up )[denom_up]
            muTrig_down[denom_dn!=0] = ( (1 - (1. - muTrig_data + muTrigerr_data).prod()) / denom_dn )[denom_dn]

            muSF = (muID*muIso).prod()*muTrig
            muSF_up = ((muID + muIDerr) * (muIso + muIsoerr) * muTrig_up).prod()
            muSF_down = ((muID - muIDerr) * (muIso - muIsoerr) * muTrig_down).prod()
            if self.debug:
                print('muID weight: ', (muID.prod()*muIso.prod())[df.event==self.test_event])
                print('Avg. muID weight: ', (muID.prod()*muIso.prod()).mean())
                print('muTrig weight: ', muTrig[df.event==self.test_event])
                print('Avg. muTrig weight: ', muTrig.mean())

            add_systematic(weights, 'muSF', muSF, muSF_up, muSF_down)


        #---------------------------------------------------------------#        
        # Loop over JEC variations and fill jet variables
        #---------------------------------------------------------------#        

        ret_jec_loop = {}
        for v_name in jet_variation_names:
             ret_jec_loop[v_name] = self.jec_loop(v_name, is_mc, df, dataset, mask, muons, mu1, mu2, muon_variables, two_muons, jets, weights, numevents)
    
        weights = ret_jec_loop['nominal']['weights']
        two_jets = ret_jec_loop['nominal']['two_jets']
        category = ret_jec_loop['nominal']['category']
        variable_map = ret_jec_loop['nominal']['variable_map']
        genJetMass = variable_map['genJetMass']
    
        if self.debug:
 #           print(df.event[mask&two_jets])
            for k,v in variable_map.items():            
                print(k, v[df.event==self.test_event])

#        if ("dy" in dataset or "ewk" in dataset) and self.do_lheweights:
#            for i in range(9):
#                try:
#                    variable_map[f'LHEScaleWeight_{i}'] = df.LHEScaleWeight[:,i]
#                except:
#                    variable_map[f'LHEScaleWeight_{i}'] = np.ones(numevents, dtype=float)
            
        for syst in weights.columns:
            variable_map[f'weight_{syst}'] = np.array(weights[syst])

        #---------------------------------------------------------------#        
        # Evaluate DNN score for each variation
        #---------------------------------------------------------------#        

        variated_scores = {}
        if self.evaluate_dnn:            
            from config.parameters import training_features
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, 
                        inter_op_parallelism_threads=1, 
                        allow_soft_placement=True,
                        device_count = {'CPU': 1})
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            sess = tf.compat.v1.Session(config=config)
            with sess:
                # BOTTLENECK: can't load model outside of a worker
                # https://github.com/keras-team/keras/issues/9964
                dnn_model = load_model(f'output/trained_models/test_{self.year}.h5')
                scaler = np.load(f"output/trained_models/scalers_{self.year}.npy")
                
                for variation in jet_variation_names:
                    if (variation != 'nominal') and (not self.do_jecunc): continue
                    if (variation != 'nominal') and not is_mc:
                        variable_map[f'dnn_score_{variation}'] = variable_map['dnn_score_nominal']
                        continue
                    dnn_score = np.full(numevents, -1.)
                    regions = get_regions(ret_jec_loop[variation]['variable_map']['dimuon_mass'])
                    for region, rcut in regions.items():
                        df_for_dnn = pd.DataFrame(columns=training_features)
                        variated_mask = ret_jec_loop[variation]['two_jets'] & rcut
                        n_rows = len(ret_jec_loop[variation]['variable_map']['dimuon_mass'][variated_mask].flatten())
                        for trf in training_features:
                            if trf=='dimuon_mass' and region!='h-peak':
                                feature_column = np.full(sum(variated_mask), 125.)
                            feature_column = ret_jec_loop[variation]['variable_map'][trf][variated_mask]
                            assert(n_rows==len(feature_column))
                            df_for_dnn[trf] = feature_column
                        df_for_dnn[training_features] = (df_for_dnn[training_features]-scaler[0])/scaler[1]
                        try:
                            prediction = dnn_model.predict(df_for_dnn[training_features], verbose=0)
                            pred_array = tf.reshape(prediction, [-1]).eval()
                        except:
                            pred_array = np.zeros(sum(variated_mask))

                        dnn_score[variated_mask] = pred_array
                    variable_map[f'dnn_score_{variation}'] = np.arctanh((dnn_score))
                    variated_scores.update({f'dnn_score_{variation}':variation})
                    
            if self.timer:
                self.timer.add_checkpoint("Evaluated DNN")
                
            if self.do_pdf and is_mc and self.year!='2016':
                pdf_rms = np.zeros(numevents, dtype=float)
                if ("dy" in dataset or "ewk" in dataset or "ggh" in dataset or "vbf" in dataset):
                    pdf_wgts = {}
                    dnn_pdf = pd.DataFrame()
                    for i in range(self.parameters["n_pdf_variations"]):
                        pdf_wgt = df.LHEPdfWeight[:,i][two_jets]
                        dnn_v = variable_map['dnn_score_nominal'][two_jets]*pdf_wgt
                        dnn_pdf = dnn_pdf.append(pd.Series(dnn_v),ignore_index=True)
                    pdf_rms[two_jets] = np.nan_to_num(2*dnn_pdf.std(axis=0).values, nan=0, posinf=0, neginf=0)
                variable_map['pdf_rms'] = pdf_rms

        #---------------------------------------------------------------#        
        # Fill outputs
        #---------------------------------------------------------------#                        

        #------------------ Binned outputs ------------------#  
        regions = get_regions(variable_map['dimuon_mass'])            

        for vname, expression in variable_map.items():
            if vname not in output['binned']: continue
            for cname in self.channels:
                ccut = (category==cname)
                for rname, rcut in regions.items():
                    if (dataset in self.overlapping_samples) and (dataset not in self.specific_samples[rname][cname]): 
                        continue

                    if vname in variated_scores.keys():
                        genJetMass = ret_jec_loop[variated_scores[vname]]['variable_map']['genJetMass']
                        rcut = get_regions(ret_jec_loop[variated_scores[vname]]['variable_map']['dimuon_mass'])[rname]
                        ccut = (ret_jec_loop[variated_scores[vname]]['category']==cname)
                        wgts = ret_jec_loop[variated_scores[vname]]['weights']
                    else:
                        genJetMass = variable_map['genJetMass']
                        wgts = weights
                        
                    if ('dy_m105_160_vbf_amc' in dataset) and ('vbf' in cname):
                        ccut = ccut & (genJetMass > 350.)
                    if ('dy_m105_160_amc' in dataset) and ('vbf' in cname):
                        ccut = ccut & (genJetMass < 350.)                        
                    value = expression[rcut & ccut]
                    if not value.size: continue # skip empty arrays
                    for syst in weights.columns:
                        if isinstance(value, awkward.JaggedArray):
                            weight = wgts[syst][rcut & ccut][value.any()]
                        else:
                            weight = wgts[syst][rcut & ccut]
#                        if ('nominal' in syst) and ('h-peak' in rname) and (vname=='dimuon_mass'):
#                            print('binned:', sum(weight))
                        output['binned'][vname].fill(**{'dataset': dataset, 'region': rname, 'channel': cname, 'syst': syst,\
                                             vname: value.flatten(), 'weight': weight})

        #----------------- Unbinned outputs -----------------#
        if self.save_unbin:
            for v in self.vars_unbin:
                if v not in variable_map: continue
                if 'dnn_score' in v: continue 
                for cname in self.channels:
                    ccut = (category==cname)
                    for rname, rcut in regions.items():
                        if ('dy_m105_160_vbf_amc' in dataset) and ('vbf' in cname):
                            ccut = ccut & (genJetMass > 350.)
                        if ('dy_m105_160_amc' in dataset) and ('vbf' in cname):
                            ccut = ccut & (genJetMass < 350.)
                        if v=='dimuon_mass':
                            output[f'event_weight_{cname}_{rname}'] +=\
                                    processor.column_accumulator(np.array(variable_map['weight_nominal'][rcut & ccut]))
#                        if  ('h-peak' in rname) and ('weight_nominal' in v):
#                            print('unbinned:', sum(variable_map[v][rcut & ccut].flatten()))
                        output['unbinned'][f'{v}_unbin_{dataset}_c_{cname}_r_{rname}'] +=\
                            processor.column_accumulator(np.array(variable_map[v][rcut & ccut].flatten()))
  #      print(output['unbinned'][f'dimuon_mass_unbin_{dataset}_c_vbf_r_z-peak'].value.shape)
  #      print(output['unbinned'][f'weight_nominal_unbin_{dataset}_c_vbf_r_z-peak'].value.shape)
        if self.timer:
            self.timer.add_checkpoint("Filled outputs")
            
        if self.timer:
            self.timer.summary()

        variable_map.clear()
        variated_scores.clear()
        for ret in ret_jec_loop.values():
            ret['variable_map'].clear()
            ret.clear()
        ret_jec_loop.clear()
        return output

    
    def jec_loop(self, variation, is_mc, df, dataset, mask, muons, mu1, mu2, muon_variables, two_muons, jets, weights, numevents):
        
        #---------------------------------------------------------------#        
        # Initialize jet-related variables
        #---------------------------------------------------------------#        
        
        jet1_variable_names = ['jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_qgl', 'jet1_id', 'jet1_puid']
        jet1_variables = {}
        for n in jet1_variable_names:
            jet1_variables[n] = np.full(numevents, -999.)

        jet2_variable_names = ['jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_qgl', 'jet2_id', 'jet2_puid']
        jet2_variables = {}
        for n in jet2_variable_names:
            jet2_variables[n] = np.full(numevents, -999.)
            
        dijet_variable_names = ['jj_mass', 'jj_pt', 'jj_eta', 'jj_phi', 'jj_dEta', 'jj_dPhi']
        dijet_variables = {}
        for n in dijet_variable_names:
            dijet_variables[n] = np.full(numevents, -999.)
            
        mmj_variable_names=['mmj1_dEta','mmj1_dPhi','mmj1_dR','mmj2_dEta','mmj2_dPhi','mmj2_dR','mmj_min_dEta', 'mmj_min_dPhi']
        mmj_variables = {}
        for n in mmj_variable_names:
            mmj_variables[n] = np.full(numevents, -999.)

        mmjj_variable_names = ['mmjj_pt', 'mmjj_eta', 'mmjj_phi', 'mmjj_mass', 'rpt', 'zeppenfeld']
        mmjj_variables = {}
        for n in mmjj_variable_names:
            mmjj_variables[n] = np.full(numevents, -999.)
            
        softjet_variable_names = ['nsoftjets2', 'nsoftjets5', 'htsoft2', 'htsoft5']
        softjet_variables = {}
        for n in softjet_variable_names:
            softjet_variables[n] = np.full(numevents, -999.)

        #---------------------------------------------------------------#        
        # Select jets for certain pT variation
        #---------------------------------------------------------------#        
            
        if is_mc and variation!='nominal':
            pt_name = f'pt_{variation}'
        else:
            pt_name = '__fast_pt'

        mujet = jets.cross(muons, nested=True)
        _,_,deltar_mujet = delta_r(mujet.i0.eta, mujet.i1.eta, mujet.i0.phi, mujet.i1.phi)
        deltar_mujet_ok =  (deltar_mujet > self.parameters["min_dr_mu_jet"]).all()      

            
        jet_selection = ((jets[pt_name] > self.parameters["jet_pt_cut"]) &\
                     (abs(jets.eta) < self.parameters["jet_eta_cut"])) & deltar_mujet_ok

        jets = jets[jet_selection]

        #---------------------------------------------------------------#        
        # Calculate PUID scale factors and apply PUID
        #---------------------------------------------------------------#        
       
        # Jet PU ID
        jet_puid_opt = self.parameters["jet_puid"]
        jet_puid_wps = {
            "loose": ( ((jets.puId >= 4) & (jets[pt_name] < 50)) | (jets[pt_name] > 50)),
            "medium": ( ((jets.puId >= 6) & (jets[pt_name] < 50)) | (jets[pt_name] > 50)),
            "tight": ( ((jets.puId >= 7) & (jets[pt_name] < 50)) | (jets[pt_name] > 50)),
        }
        if jet_puid_opt in ["loose", "medium", "tight"]:
            jet_puid = jet_puid_wps[jet_puid_opt]
        elif "2017corrected" in jet_puid_opt:
            eta_window = ((abs(jets.eta)>2.6)&(abs(jets.eta)<3.0))
            not_eta_window = ((abs(jets.eta)<2.6)|(abs(jets.eta)>3.0))
            jet_puid = (eta_window & (jets.puId >= 7)) | (not_eta_window & jet_puid_wps['loose'])
        else:
            jet_puid = jets.ones_like()

        if self.debug:
            print("Jets passing PUID: ", jet_puid.flatten().sum())            

        # Jet PUID scale factors
        if is_mc:     
            puid_weight = self.get_puid_weight(jets, pt_name, jet_puid_opt, jet_puid, numevents)   
            weights = weights.multiply(puid_weight, axis=0)

            if self.debug:
                print('Jet PUID weight: ', puid_weight[df.event==self.test_event])
                print("Avg. jet PU ID weight: ", puid_weight.mean())            

        jets = jets[jet_puid]              

        #---------------------------------------------------------------#        
        # Fill jet-related variables
        #---------------------------------------------------------------#        
        one_jet = ((jet_selection).any() & (jets.counts>0))
        two_jets = ((jet_selection).any() & (jets.counts>1))

        cols = {'pt':pt_name,
                'eta':'__fast_eta',
                'phi':'__fast_phi',
                'mass':'__fast_mass'}
        cols.update({k:k for k in ['qgl','jetId','puId', 'btagDeepB']})
        if is_mc:
            cols.update({k:k for k in ['genJetIdx','ptGenJet','hadronFlavour']})

        jet1_mask = one_jet.astype(int)
        jet2_mask = two_jets.astype(int)

        jetarrays = {key:jets[v].flatten() for key, v in cols.items()}
        jet1_arrays = {key:jets[one_jet][:,0][v].flatten() for key, v in cols.items()}
        jet2_arrays = {key:jets[two_jets][:,1][v].flatten() for key, v in cols.items()}

        jet1 = JaggedCandidateArray.candidatesfromcounts(jet1_mask, **jet1_arrays)
        jet2 = JaggedCandidateArray.candidatesfromcounts(jet2_mask, **jet2_arrays)
        jets = JaggedCandidateArray.candidatesfromcounts(jets.counts, **jetarrays) 
        
        jetarrays.clear()
        jet1_arrays.clear()
        jet2_arrays.clear()
        
        jet1_variables['jet1_pt'][one_jet] = jet1['__fast_pt'].flatten()
        jet1_variables['jet1_eta'][one_jet] = jet1['__fast_eta'].flatten()
        jet1_variables['jet1_phi'][one_jet] = jet1['__fast_phi'].flatten()
        jet1_variables['jet1_qgl'][one_jet] = jet1.qgl.flatten()
        jet1_variables['jet1_id'][one_jet] = jet1.jetId.flatten()
        jet1_variables['jet1_puid'][one_jet] = jet1.puId.flatten()
        
        jet2_variables['jet2_pt'][two_jets] = jet2['__fast_pt'].flatten()
        jet2_variables['jet2_eta'][two_jets] = jet2['__fast_eta'].flatten()
        jet2_variables['jet2_phi'][two_jets] = jet2['__fast_phi'].flatten()
        jet2_variables['jet2_qgl'][two_jets] = jet2.qgl.flatten()
        jet2_variables['jet2_id'][two_jets] = jet2.jetId.flatten()
        jet2_variables['jet2_puid'][two_jets] = jet2.puId.flatten()
                        
        dijet_variables['jj_pt'][two_jets],\
        dijet_variables['jj_eta'][two_jets],\
        dijet_variables['jj_phi'][two_jets],\
        dijet_variables['jj_mass'][two_jets] = p4_sum(jet1[two_jets], jet2[two_jets])
        dijet_variables['jj_dEta'][two_jets] = abs(jet1_variables['jet1_eta'][two_jets]-jet2_variables['jet2_eta'][two_jets])
        dijet_variables['jj_dPhi'][two_jets] = abs(jet1[two_jets].p4.delta_phi(jet2[two_jets].p4))

        # Definition with rapidity would be different
        mmjj_variables['zeppenfeld'][two_muons&two_jets] = (muon_variables['dimuon_eta'][two_muons&two_jets] - 0.5*\
                          (jet1_variables['jet1_eta'][two_muons&two_jets] + jet2_variables['jet2_eta'][two_muons&two_jets]))
        
        mmjj_variables['mmjj_pt'][two_muons&two_jets],\
        mmjj_variables['mmjj_eta'][two_muons&two_jets],\
        mmjj_variables['mmjj_phi'][two_muons&two_jets],\
        mmjj_variables['mmjj_mass'][two_muons&two_jets] = p4_sum_alt(muon_variables['dimuon_pt'][two_muons&two_jets],\
                                                          muon_variables['dimuon_eta'][two_muons&two_jets],\
                                                          muon_variables['dimuon_phi'][two_muons&two_jets],\
                                                          muon_variables['dimuon_mass'][two_muons&two_jets],\
                                                          dijet_variables['jj_pt'][two_muons&two_jets],\
                                                          dijet_variables['jj_eta'][two_muons&two_jets],\
                                                          dijet_variables['jj_phi'][two_muons&two_jets],\
                                                          dijet_variables['jj_mass'][two_muons&two_jets])

        mmjj_variables['rpt'][two_muons&two_jets] = mmjj_variables['mmjj_pt'][two_muons&two_jets] / \
                                                    (muon_variables['dimuon_pt'][two_muons&two_jets] + \
                                                    jet1_variables['jet1_pt'][two_muons&two_jets] + \
                                                     jet2_variables['jet2_pt'][two_muons&two_jets])

        mmj_variables['mmj1_dEta'][two_muons&one_jet],\
        mmj_variables['mmj1_dPhi'][two_muons&one_jet],\
        mmj_variables['mmj1_dR'][two_muons&one_jet] = delta_r(muon_variables['dimuon_eta'][two_muons&one_jet].flatten(),\
                                                      jet1_variables['jet1_eta'][two_muons&one_jet].flatten(),\
                                                      muon_variables['dimuon_phi'][two_muons&one_jet].flatten(),\
                                                      jet1_variables['jet1_phi'][two_muons&one_jet].flatten())
            
        mmj_variables['mmj2_dEta'][two_muons&two_jets],\
        mmj_variables['mmj2_dPhi'][two_muons&two_jets],\
        mmj_variables['mmj2_dR'][two_muons&two_jets] = delta_r(muon_variables['dimuon_eta'][two_muons&two_jets].flatten(),\
                                                       jet2_variables['jet2_eta'][two_muons&two_jets].flatten(),\
                                                       muon_variables['dimuon_phi'][two_muons&two_jets].flatten(),\
                                                       jet2_variables['jet2_phi'][two_muons&two_jets].flatten())

        mmj_variables['mmj_min_dEta'] = np.where(mmj_variables['mmj1_dEta'],\
                                                 mmj_variables['mmj2_dEta'],\
                                                 (mmj_variables['mmj1_dEta'] < mmj_variables['mmj2_dEta']))
        
        mmj_variables['mmj_min_dEta'] = np.where(mmj_variables['mmj1_dPhi'],\
                                                 mmj_variables['mmj2_dPhi'],\
                                                 (mmj_variables['mmj1_dPhi'] < mmj_variables['mmj2_dPhi']))

        #---------------------------------------------------------------#        
        # Fill soft activity jet variables
        #---------------------------------------------------------------#        
        
        sjarrays = {key:df.SoftActivityJet[key].flatten() for key in df.SoftActivityJet.columns}   
        sjarrays.update({'mass':0})
        softjets = JaggedCandidateArray.candidatesfromcounts(df.SoftActivityJet.counts, **sjarrays)
        
        softjet_variables['nsoftjets2'],\
        softjet_variables['htsoft2'] = self.get_softjet_vars(2, df, muons, softjets, jet1, jet2, one_jet, two_jets)
        
        softjet_variables['nsoftjets5'],\
        softjet_variables['htsoft5'] = self.get_softjet_vars(5, df, muons, softjets, jet1, jet2, one_jet, two_jets)
        
        #---------------------------------------------------------------#        
        # Calculate getJetMass
        #---------------------------------------------------------------#        
        
        genJetMass = np.zeros(numevents, dtype=float)        
        if is_mc:     
            genJetMass_filtered = genJetMass[two_jets]
            idx1 = jet1.genJetIdx[two_jets].flatten()
            idx2 = jet2.genJetIdx[two_jets].flatten()
            gjets = df.GenJet[two_jets]
            has_gen_pair = (idx1 >= 0) & (idx2 >= 0) & (idx1 < gjets.counts) & (idx2 < gjets.counts)
            _,_,_, genJetMass_filtered[has_gen_pair] = p4_sum(gjets[has_gen_pair,idx1[has_gen_pair]],\
                                                               gjets[has_gen_pair,idx2[has_gen_pair]])
            genJetMass[two_jets] = genJetMass_filtered
            
        #---------------------------------------------------------------#        
        # Apply remaining cuts
        #---------------------------------------------------------------#        
        
        leading_jet_pt = np.zeros(numevents, dtype=bool)
        leading_jet_pt[jets.counts>0] = (jets[jets.counts>0][:,0][[pt_name]]>35.)
        mask = mask & leading_jet_pt       

        vbf_cut = (dijet_variables['jj_mass']>400)&(dijet_variables['jj_dEta']>2.5)
        
        #---------------------------------------------------------------#        
        # Calculate btag SF and apply btag veto
        #---------------------------------------------------------------#        
        mass = (muon_variables['dimuon_mass']>115)&(muon_variables['dimuon_mass']<135)

        bjet_sel_mask = mask&two_jets&vbf_cut
        
        # Btag weight
        if self.do_btagsf and is_mc:
            btag_wgt = self.get_btag_weight(jets, pt_name, weights, bjet_sel_mask, numevents)
            weights = weights.multiply(btag_wgt, axis=0)
            
        # Separate from ttH and VH phase space        
        nBtagLoose = jets[(jets.btagDeepB>self.parameters["btag_loose_wp"]) & (abs(jets.eta)<2.5)].counts
        nBtagMedium = jets[(jets.btagDeepB>self.parameters["btag_medium_wp"])  & (abs(jets.eta)<2.5)].counts
        mask = mask & (nBtagLoose<2) & (nBtagMedium<1)

        if self.debug:
            print("VBF category (unweighted):", sum(mask&two_jets&vbf_cut))
            print("VBF category (weighted): ", weights['nominal'][mask&two_jets&vbf_cut].sum())

        #---------------------------------------------------------------#        
        # Define categories
        #---------------------------------------------------------------#        
        
        category = np.empty(df.shape[0], dtype=object)
        category[mask&(~two_jets)] = 'ggh_01j'
        category[mask&two_jets&(~vbf_cut)] = 'ggh_2j'
        category[mask&two_jets&vbf_cut] = 'vbf'
#        category[mask&two_jets] = 'vbf'         

        #---------------------------------------------------------------#        
        # Fill outputs
        #---------------------------------------------------------------#        
            
        ret = {}
        ret.update(**muon_variables)
        ret.update(**dijet_variables)
        ret.update(**jet1_variables)
        ret.update(**jet2_variables)
        ret.update(**dijet_variables)
        ret.update(**mmjj_variables)
        ret.update(**mmj_variables)
        ret.update(**softjet_variables)
        ret.update(**{
                'njets': jets.counts.flatten(),
                'npv': df.PV.npvsGood.flatten(),
                'met': df.MET.pt.flatten(),
                'genJetMass': genJetMass,
                'event': df.event.flatten(),
            })


        return {
                'variable_map': ret, 
                'category': category,
                'two_jets': two_jets, 
                'weights': weights, 
               }
    
    def get_puid_weight(self, jets, pt_name, jet_puid_opt, jet_puid, numevents):
        if "2017corrected" in jet_puid_opt:
            h_eff_name_L = f"h2_eff_mc{self.year}_L"
            h_sf_name_L = f"h2_eff_sf{self.year}_L"
            h_eff_name_T = f"h2_eff_mc{self.year}_T"
            h_sf_name_T = f"h2_eff_sf{self.year}_T"
            puid_eff_L = self.evaluator[h_eff_name_L](jets[pt_name], jets.eta)
            puid_sf_L = self.evaluator[h_sf_name_L](jets[pt_name], jets.eta)
            puid_eff_T = self.evaluator[h_eff_name_T](jets[pt_name], jets.eta)
            puid_sf_T = self.evaluator[h_sf_name_T](jets[pt_name], jets.eta)

            jets_passed_L = (jets[pt_name]>25) & (jets[pt_name]<50) & jet_puid & ((abs(jets.eta)<2.6)|(abs(jets.eta)>3.0))
            jets_failed_L = (jets[pt_name]>25) & (jets[pt_name]<50) & (~jet_puid) & ((abs(jets.eta)<2.6)|(abs(jets.eta)>3.0))
            jets_passed_T = (jets[pt_name]>25) & (jets[pt_name]<50) & jet_puid & ((abs(jets.eta)>2.6)&(abs(jets.eta)<3.0))
            jets_failed_T = (jets[pt_name]>25) & (jets[pt_name]<50) & (~jet_puid) & ((abs(jets.eta)>2.6)&(abs(jets.eta)<3.0))

            pMC_L   = puid_eff_L[jets_passed_L].prod() * (1.-puid_eff_L[jets_failed_L]).prod() 
            pMC_T   = puid_eff_T[jets_passed_T].prod() * (1.-puid_eff_T[jets_failed_T]).prod() 

            pData_L = puid_eff_L[jets_passed_L].prod() * puid_sf_L[jets_passed_L].prod() * \
            (1. - puid_eff_L[jets_failed_L] * puid_sf_L[jets_failed_L]).prod()
            pData_T = puid_eff_T[jets_passed_T].prod() * puid_sf_T[jets_passed_T].prod() * \
            (1. - puid_eff_T[jets_failed_T] * puid_sf_T[jets_failed_T]).prod()

            puid_weight = np.ones(numevents)
            puid_weight[pMC_L*pMC_T!=0] = np.divide((pData_L*pData_T)[pMC_L*pMC_T!=0], (pMC_L*pMC_T)[pMC_L*pMC_T!=0])

        else:
            wp_dict = {"loose": "L", "medium": "M", "tight": "T"}
            wp = wp_dict[jet_puid_opt]
            h_eff_name = f"h2_eff_mc{self.year}_{wp}"
            h_sf_name = f"h2_eff_sf{self.year}_{wp}"
            puid_eff = self.evaluator[h_eff_name](jets[pt_name], jets.eta)
            puid_sf = self.evaluator[h_sf_name](jets[pt_name], jets.eta)
            jets_passed = (jets[pt_name]>25) & (jets[pt_name]<50) & jet_puid
            jets_failed = (jets[pt_name]>25) & (jets[pt_name]<50) & (~jet_puid)  

            pMC   = puid_eff[jets_passed].prod() * (1.-puid_eff[jets_failed]).prod() 
            pData = puid_eff[jets_passed].prod() * puid_sf[jets_passed].prod() * \
            (1. - puid_eff[jets_failed] * puid_sf[jets_failed]).prod()
            puid_weight = np.ones(numevents)
            puid_weight[pMC!=0] = np.divide(pData[pMC!=0], pMC[pMC!=0])  

        return puid_weight
    
    def get_btag_weight(self, jets, pt_name, weights, bjet_sel_mask, numevents):
        btag_wgt = np.ones(numevents, dtype=float)
        jets_ = jets[abs(jets.eta)<2.4]
        jet_pt_ = awkward.JaggedArray.fromcounts(jets_[jets_.counts>0].counts, np.minimum(jets_[pt_name].flatten(), 1000.))
        btag_wgt[(jets_.counts>0)] = self.btag_sf('central', jets_[jets_.counts>0].hadronFlavour,\
                                                  abs(jets_[jets_.counts>0].eta), jet_pt_,\
                                                  jets_[jets_.counts>0].btagDeepB, True).prod()
        btag_wgt[btag_wgt<0.01] = 1.

        sum_before = weights['nominal'][bjet_sel_mask].sum()
        sum_after = weights['nominal'][bjet_sel_mask].multiply(btag_wgt[bjet_sel_mask], axis=0).sum()
        btag_wgt = btag_wgt*sum_before/sum_after
        return btag_wgt

        
    def get_softjet_vars(self, cutoff, df, muons, softjets, jet1, jet2, one_jet, two_jets):
        nsoftjets = df[f'SoftActivityJetNjets{cutoff}']
        htsoft = df[f'SoftActivityJetHT{cutoff}']
        muons = muons[:, 0:2]
        # Events with 0 selected jets: clear soft jets from muons
        no_jets = ~one_jet
        sj_mm0 = softjets[no_jets].cross(muons[no_jets], nested=True)
        _,_,dr_mm0 = delta_r(sj_mm0.i0.eta, sj_mm0.i1.eta, sj_mm0.i0.phi, sj_mm0.i1.phi)
        bad_sj_mm0 = (dr_mm0*dr_mm0 < self.parameters["softjet_dr2"])
        bad_sj0 = (bad_sj_mm0.any()).all()

        sj_sel0 = np.zeros(df.shape[0], dtype=bool)
        sj_sel0[no_jets] = ((softjets.pt[no_jets]>cutoff) & bad_sj0).any()
        nsoftjets[sj_sel0] = nsoftjets[sj_sel0] - softjets[sj_sel0].counts
        htsoft[sj_sel0] = htsoft[sj_sel0] - softjets.pt[sj_sel0].sum()

        # Events with exactly 1 selected jet: clear soft jets from muons and the jet
        only_one_jet = one_jet & (~two_jets)

        sj_mm1 = softjets[only_one_jet].cross(muons[only_one_jet], nested=True)
        _,_,dr_mm1 = delta_r(sj_mm1.i0.eta, sj_mm1.i1.eta, sj_mm1.i0.phi, sj_mm1.i1.phi)
        bad_sj_mm1 = (dr_mm1*dr_mm1 < self.parameters["softjet_dr2"])

        sj_j1_1 = softjets[only_one_jet].cross(jet1[only_one_jet], nested=True)
        _,_,dr_j1_1 = delta_r(sj_j1_1.i0.eta, sj_j1_1.i1.eta, sj_j1_1.i0.phi, sj_j1_1.i1.phi)
        bad_sj_j1_1 = (dr_j1_1*dr_j1_1 < self.parameters["softjet_dr2"])

        bad_sj1 = (bad_sj_mm1.any() | bad_sj_j1_1).all()

        sj_sel1 = np.zeros(df.shape[0], dtype=bool)
        sj_sel1[only_one_jet] = ((softjets.pt[only_one_jet]>cutoff) & bad_sj1).any()
        nsoftjets[sj_sel1] = nsoftjets[sj_sel1] - softjets[sj_sel1].counts
        htsoft[sj_sel1] = htsoft[sj_sel1] - softjets.pt[sj_sel1].sum()

        # Events with two selected jets: clear soft jets from muons and selected jets
        
        sj_mm2 = softjets[two_jets].cross(muons[two_jets], nested=True)
        _,_,dr_mm2 = delta_r(sj_mm2.i0.eta, sj_mm2.i1.eta, sj_mm2.i0.phi, sj_mm2.i1.phi)
        bad_sj_mm2 = (dr_mm2*dr_mm2 < self.parameters["softjet_dr2"])

        sj_j1_2 = softjets[two_jets].cross(jet1[two_jets], nested=True)
        _,_,dr_j1_2 = delta_r(sj_j1_2.i0.eta, sj_j1_2.i1.eta, sj_j1_2.i0.phi, sj_j1_2.i1.phi)
        bad_sj_j1_2 = (dr_j1_2*dr_j1_2 < self.parameters["softjet_dr2"])

        sj_j2 = softjets[two_jets].cross(jet2[two_jets], nested=True)
        _,_,dr_j2 = delta_r(sj_j2.i0.eta, sj_j2.i1.eta, sj_j2.i0.phi, sj_j2.i1.phi)
        bad_sj_j2 = (dr_j2*dr_j2 < self.parameters["softjet_dr2"])

        bad_sj = (bad_sj_mm2.any() | bad_sj_j1_2 | bad_sj_j2).all()

        eta_sj_j1 = (sj_j1_2.i0.eta > sj_j1_2.i1.eta).all()
        eta_sj_j2 = (sj_j2.i0.eta > sj_j2.i1.eta).all()
        eta_j1j2 = (sj_j1_2.i1.eta > sj_j2.i1.eta).all()

        eta_sel = ( (eta_sj_j1 | ~eta_sj_j2) & eta_j1j2 ) | ( (~eta_sj_j1 | eta_sj_j2) & ~eta_j1j2 )

        sj_sel2 = np.zeros(df.shape[0], dtype=bool)
        sj_sel2[two_jets] = (((softjets.pt[two_jets]>cutoff) | eta_sel) & bad_sj).any()
        nsoftjets[sj_sel2] = nsoftjets[sj_sel2] - softjets[sj_sel2].counts
        htsoft[sj_sel2] = htsoft[sj_sel2] - softjets.pt[sj_sel2].sum()
        
        return nsoftjets, htsoft
  
    def postprocess(self, accumulator):
        return accumulator
