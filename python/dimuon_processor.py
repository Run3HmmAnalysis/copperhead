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
import pandas as pd

from python.utils import p4_sum, p4_sum_alt, delta_r
from python.timer import Timer
from python.samples_info import SamplesInfo
from python.weights import Weights
from python.corrections import musf_lookup, musf_evaluator, pu_lookup, pu_evaluator, NNLOPS_Evaluator, roccor_evaluator
from python.corrections import qgl_weights, puid_weights, btag_weights, geofit_evaluator, fsr_evaluator

import gc
    
def get_regions(mass):
    regions = {
        "z-peak": ((mass>76) & (mass<106)),
        "h-sidebands": ((mass>110) & (mass<115)) | ((mass>135) & (mass<150)),
        "h-peak": ((mass>115) & (mass<135)),
    }
    return regions

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

class DimuonProcessor(processor.ProcessorABC):
    def __init__(self, samp_info, evaluate_dnn=False,\
                 do_timer=False, save_unbin=True, do_lheweights=False, do_jec=True,\
                 do_jer=True, do_jecunc=False, do_pdf=True, debug=False): 
        from config.parameters import parameters
        from config.variables import variables
        if not samp_info:
            print("Samples info missing!")
            return
        self.samp_info = samp_info
        self.year = self.samp_info.year
        self.debug = debug
        self.mass_window = [76, 150]
        self.save_unbin = save_unbin
        self.do_jec = do_jec
        self.do_jer = do_jer
        self.do_jecunc = do_jecunc
        self.do_pdf = do_pdf
        self.evaluate_dnn = evaluate_dnn
        self.do_lheweights = do_lheweights
        self.parameters = {k:v[self.year] for k,v in parameters.items()}

        self.timer = Timer('global') if do_timer else None
 
        self._columns = self.parameters["proc_columns"]
                        
        self.regions = self.samp_info.regions
        self.channels = self.samp_info.channels

        self.overlapping_samples = self.samp_info.overlapping_samples
        self.specific_samples = self.samp_info.specific_samples
        self.datasets_to_save_unbin = self.samp_info.datasets_to_save_unbin
        self.lumi_weights = self.samp_info.lumi_weights
        
        from config.variables import Variable
        self.variations = ['nominal', 'pu_weight', 'muSF', 'l1prefiring_weight', 'qgl_weight']
        
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
                    variables.append(Variable(f"dnn_score_{v_name}_up", f"dnn_score_{v_name}_up", 12, 0,\
                                              self.parameters["dnn_max"]))
                    variables.append(Variable(f"dnn_score_{v_name}_down", f"dnn_score_{v_name}_down", 12, 0,\
                                              self.parameters["dnn_max"]))
                
#        for i in range(9):
#            variables.append(Variable(f"LHEScaleWeight_{i}", f"LHEScaleWeight_{i}", 1, 0, 1))
        
        if self.do_pdf:
            variables.append(Variable("pdf_rms", "pdf_rms", 1, 0, 1))
        
        self.vars_unbin = [v.name for v in variables]
        
        dataset_axis = hist.Cat("dataset", "")
        region_axis = hist.Cat("region", "") # Z-peak, Higgs SB, Higgs peak
        channel_axis = hist.Cat("channel", "") # ggh or VBF  
        syst_axis = hist.Cat("syst", "")
        
        ### Prepare accumulators for binned output ###
        bin_dict = {}
        for v in variables:
            if v.name=='dimuon_mass':
                axis = hist.Bin(v.name, v.caption, v.nbins, self.mass_window[0], self.mass_window[1])
            else:
                axis = hist.Bin(v.name, v.caption, v.nbins, v.xmin, v.xmax)
            bin_dict[v.name] = hist.Hist("Counts", dataset_axis, region_axis, channel_axis, syst_axis, axis)  
        accumulator_binned = processor.dict_accumulator(bin_dict)
        
        ### Prepare accumulators for unbinned output ###
        unbin_dict = {}
        if self.save_unbin:
            for p in self.samp_info.samples:
                for v in self.vars_unbin:
                    if 'dnn_score' in v: continue
                    for c in self.channels:
                        for r in self.regions:
                            unbin_dict[f'{v}_unbin_{p}_c_{c}_r_{r}'] = processor.column_accumulator(np.ndarray([]))
                            # have to encode everything into the name because having multiple axes isn't possible
        accumulator_unbinned = processor.dict_accumulator(unbin_dict)
        ### --------------------------------------- ###
        
        acc_dicts = {'binned':accumulator_binned, 'unbinned':accumulator_unbinned}        
        accumulators = processor.dict_accumulator(acc_dicts)
        self._accumulator = accumulators
        ### --------------------------------------- ###
        
        ### Prepare lookups for corrections ###        
        rochester_data = txt_converters.convert_rochester_file(self.parameters["roccor_file"], loaduncs=True)
        self.roccor_lookup = rochester_lookup.rochester_lookup(rochester_data)
        self.musf_lookup = musf_lookup(self.parameters)
        self.pu_lookup = pu_lookup(self.parameters)
        self.pu_lookup_up = pu_lookup(self.parameters, 'up')
        self.pu_lookup_down = pu_lookup(self.parameters, 'down')
        
        self.btag_lookup = BTagScaleFactor(self.parameters["btag_sf_csv"], BTagScaleFactor.RESHAPE,\
                                       'iterativefit,iterativefit,iterativefit')
        
        
        ### Prepare evaluator for corrections that can be loaded together ###        
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
        
        ### Prepare evaluators for JEC, JER and their systematics ### 
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
        


    @property
    def accumulator(self):
        return self._accumulator
    
    @property
    def columns(self):
        return self._columns
    
    def process(self, df):
        #---------------------------------------------------------------#        
        # Filter out events not passing HLT or having less than 2 muons.
        #---------------------------------------------------------------#
        self.test_event = 71548460
        self.debug=False
        self.timer=None
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

        weights = Weights(df)
        
        if is_mc:    
            mask = np.ones(numevents, dtype=bool)
            
            #---------------------------------------------------------------#        
            # Apply gen.weights, pileup weights, lumi weights, L1 prefiring weights
            # 
            #---------------------------------------------------------------# 
            
            genweight = df.genWeight.flatten()
            weights.add_weight('genweight', genweight)    

            pu_weight = pu_evaluator(self.pu_lookup, numevents, df.Pileup.nTrueInt)
            pu_weight_up = pu_evaluator(self.pu_lookup_up, numevents, df.Pileup.nTrueInt)
            pu_weight_down = pu_evaluator(self.pu_lookup_down, numevents, df.Pileup.nTrueInt)
            weights.add_weight_with_variations('pu_weight', pu_weight, pu_weight_up, pu_weight_down)

            
            if dataset in self.lumi_weights:
                weights.add_weight('lumi', self.lumi_weights[dataset])
             
            
            if self.parameters["do_l1prefiring_wgts"]:
                prefiring_wgt = df.L1PreFiringWeight.Nom.flatten()
                weights.add_weight_with_variations('l1prefiring_weight',\
                                                   prefiring_wgt,\
                                                   df.L1PreFiringWeight.Up.flatten(),\
                                                   df.L1PreFiringWeight.Dn.flatten())
                
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
            
        mu = df.Muon[df.Muon.pt>9]
        muons_pt = mu.pt.flatten()
        muons_pt_raw = muons_pt
        muons_eta = mu.eta.flatten()
        muons_eta_raw = muons_eta
        muons_phi = mu.phi.flatten()
        muons_mass = mu.mass.flatten()
        muons_iso = mu.pfRelIso04_all.flatten()          
        has_fsr = np.zeros(len(mu.pt.flatten()), dtype=bool)
        
        if True: # do Roch
            roch_corr = roccor_evaluator(self.roccor_lookup, is_mc, mu).flatten()
        else:
            roch_corr = np.ones(len(muons_pt), dtype=float)

        muons_pt = muons_pt*roch_corr
        muons_pt_raw = muons_pt # Rochester should be still applied

        if True: # do FSR
            fsr = df.FsrPhoton
            fsr_offsets = fsr.counts2offsets(fsr.counts)
            muons_offsets = mu.counts2offsets(mu.counts)
            fsr_pt = np.array(fsr.pt.flatten(), dtype=float)
            fsr_eta = np.array(fsr.eta.flatten(), dtype=float)
            fsr_phi = np.array(fsr.phi.flatten(), dtype=float)
            fsr_iso = np.array(fsr.relIso03.flatten(), dtype=float)
            fsr_drEt2 = np.array(fsr.dROverEt2.flatten(), dtype=float)
            
            muons_fsrIndex = np.array(mu.fsrPhotonIdx.flatten(), dtype=int)

            muons_pt, muons_eta, muons_phi, muons_mass, muons_iso, has_fsr =\
            fsr_evaluator(muons_offsets, fsr_offsets,\
                                  np.array(muons_pt, dtype=float), np.array(muons_pt_raw, dtype=float),\
                                  np.array(muons_eta, dtype=float), np.array(muons_phi, dtype=float),\
                                  np.array(muons_mass, dtype=float), np.array(muons_iso, dtype=float),\
                                  np.array(muons_fsrIndex, dtype=int), fsr_pt, fsr_eta, fsr_phi, fsr_iso, fsr_drEt2, has_fsr) 
            muons_pt_fsr = muons_pt
            
        # GeoFit correction
        if 'dxybs' in mu.columns:
            muons_dxybs = mu.dxybs.flatten()
            muons_charge = mu.charge.flatten()
            muons_pt = geofit_evaluator(muons_pt, muons_eta, muons_dxybs, muons_charge, self.year, has_fsr).flatten() 
            
        updated_attrs = {'pt': muons_pt, 'pt_raw': muons_pt_raw, 'pt_fsr': muons_pt_fsr, 'eta':muons_eta,\
                         'eta_raw':muons_eta_raw, 'phi':muons_phi, 'mass':muons_mass, 'pfRelIso04_all':muons_iso}
        
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
            
        muons = muons[(muons.pt_fsr > self.parameters["muon_pt_cut"]) &\
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
        pass_leading_pt[muons.counts>0] = (mu1.pt_fsr>self.parameters["muon_leading_pt"]).flatten()
        
        # All L3 trigger muons
        trigmuarrays = {key:df.TrigObj[key].flatten() for key in df.TrigObj.columns}
        trigmuarrays.update({'mass':0})
        trig_muons = JaggedCandidateArray.candidatesfromcounts(df.TrigObj.counts, **trigmuarrays)
        trig_muons = trig_muons[trig_muons.id == 13]
        trigmuarrays.clear()
        
        # Muons that pass tight id and iso as well as leading muon pT cut     
        mu_for_trigmatch = muons[(muons.pt_fsr > self.parameters["muon_leading_pt"]) &\
                                   (muons.pfRelIso04_all < self.parameters["muon_trigmatch_iso"]) &\
                                   muons[self.parameters["muon_trigmatch_id"]]]
        
        # For every such muon check if there is a L3 object within dR<0.1
        muTrig = mu_for_trigmatch.cross(trig_muons, nested = True)
        _,_,dr = delta_r(muTrig.i0.eta_raw, muTrig.i1.eta, muTrig.i0.phi, muTrig.i1.phi)
        has_matched_trigmuon = (dr < self.parameters["muon_trigmatch_dr"]).any()
        
        # Events where there is a trigger object matched to a tight-ID tight-Iso muon passing leading pT cut
        event_passing_trig_match = (mu_for_trigmatch[has_matched_trigmuon].counts>0).flatten()
        
        mask = mask & pass_leading_pt# & event_passing_trig_match
        
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
                    'mass':'__fast_mass',
                    'matched_muons': 'matched_muons'}
            cols.update(**{k:k for k in ['qgl','btagDeepB','ptRaw','massRaw','rho','area','jetId','puId']})
            if self.year=="2017":
                cols.update(**{'puId17':'puId17'})
            if is_mc:
                cols.update({k:k for k in ['genJetIdx','ptGenJet','hadronFlavour', 'partonFlavour']})
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
                for run in self.data_runs: # 'A', 'B', 'C', 'D', etc...
                    if run in dataset: # dataset name is something like 'data_B'
                        self.Jet_transformer_data[run].transform(jets, forceStochastic=False) 
                        print(run)
                        print(jets.pt[jets.pt<0].sum().sum())

        jets.add_attributes(**{'pt_nominal': jets.pt})
        jets = jets[jets.pt.argsort()]  

        #---------------------------------------------------------------#        
        # Apply jetID
        #---------------------------------------------------------------#        

        if "loose" in self.parameters["jet_id"]:
            jet_id = (jets.jetId >= 1)
        elif "tight" in self.parameters["jet_id"]:
            if '2016' in self.year:
                jet_id = (jets.jetId >= 3)
            else:
                jet_id = (jets.jetId >= 2)
        else:
            jet_id = jets.ones_like()

            
        good_jet_id = jet_id & (jets.qgl > -2)
        
        jets = jets[good_jet_id]         
        
        
        #---------------------------------------------------------------#        
        # Calculate other event weights
        #---------------------------------------------------------------#        
        
        if is_mc:
            if ('ggh' in dataset):
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
                weights.add_weight('nnlops', nnlopsw)

#            if 'dy' in dataset:
#                zpt_weight = np.ones(numevents, dtype=float)
#                zpt_weight[two_muons] = self.evaluator[self.zpt_path](dimuon_variables['dimuon_pt'][two_muons]).flatten()
#                weights.add_weight('zpt_weight', zpt_weight)
            
            muSF, muSF_up, muSF_down = musf_evaluator(self.musf_lookup, self.year, numevents, muons)
            weights.add_weight_with_variations('muSF', muSF, muSF_up, muSF_down)


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
            for k,v in variable_map.items():            
                print(k, v[df.event==self.test_event])

#        if ("dy" in dataset or "ewk" in dataset) and self.do_lheweights:
#            for i in range(9):
#                try:
#                    variable_map[f'LHEScaleWeight_{i}'] = df.LHEScaleWeight[:,i]
#                except:
#                    variable_map[f'LHEScaleWeight_{i}'] = np.ones(numevents, dtype=float)
            
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
                    for syst in weights.df.columns:
                        weight = wgts.get_weight(syst, rcut & ccut)
                        if len(weight)==0:
                            continue
                        output['binned'][vname].fill(**{'dataset': dataset, 'region': rname, 'channel': cname, 'syst': syst,\
                                             vname: value.flatten(), 'weight': weight})

        for syst in weights.df.columns:
            variable_map[f'weight_{syst}'] = weights.get_weight(syst)                        
                        
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
#                        if v=='dimuon_mass':
#                            output[f'event_weight_{cname}_{rname}'] +=\
#                                    processor.column_accumulator(np.array(variable_map['weight_nominal'][rcut & ccut]))
                        output['unbinned'][f'{v}_unbin_{dataset}_c_{cname}_r_{rname}'] +=\
                            processor.column_accumulator(np.array(variable_map[v][rcut & ccut].flatten()))
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
        match_mu = jets.matched_muons
        deltar_mujet_ok = ((match_mu.pfRelIso04_all>0.25) | (~match_mu.mediumId) | (match_mu.pt<20)).all()
            
        jet_selection = ((jets[pt_name] > self.parameters["jet_pt_cut"]) &\
                     (abs(jets.eta) < self.parameters["jet_eta_cut"])) & deltar_mujet_ok

        jets = jets[jet_selection]

        #---------------------------------------------------------------#        
        # Calculate PUID scale factors and apply PUID
        #---------------------------------------------------------------#        
       
        # Jet PU ID
        jet_puid_opt = self.parameters["jet_puid"]
        puId = jets.puId17 if self.year=="2017" else jets.puId
        jet_puid_wps = {
            "loose": ( ((puId >= 4) & (jets[pt_name] < 50)) | (jets[pt_name] > 50)),
            "medium": ( ((puId >= 6) & (jets[pt_name] < 50)) | (jets[pt_name] > 50)),
            "tight": ( ((puId >= 7) & (jets[pt_name] < 50)) | (jets[pt_name] > 50)),
        }
        if jet_puid_opt in ["loose", "medium", "tight"]:
            jet_puid = jet_puid_wps[jet_puid_opt]
        elif "2017corrected" in jet_puid_opt:
            eta_window = ((abs(jets.eta)>2.6)&(abs(jets.eta)<3.0))
            not_eta_window = ((abs(jets.eta)<2.6)|(abs(jets.eta)>3.0))
            jet_puid = (eta_window & (puId >= 7)) | (not_eta_window & jet_puid_wps['loose'])
        else:
            jet_puid = jets.ones_like()

        # Jet PUID scale factors
        if is_mc:     
            puid_weight = puid_weights(self.evaluator, self.year, jets, pt_name, jet_puid_opt, jet_puid, numevents)   
            weights.add_weight('puid_weight', puid_weight)

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
            cols.update({k:k for k in ['genJetIdx','ptGenJet','hadronFlavour', 'partonFlavour']})

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

        if is_mc:
            qgl_wgt = np.ones(numevents, dtype=float)
            isHerwig = ('herwig' in dataset)
            qgl_wgt[one_jet] = qgl_wgt[one_jet]*qgl_weights(jet1, isHerwig)
            qgl_wgt[two_jets] = qgl_wgt[two_jets]*qgl_weights(jet2, isHerwig)
            weights.add_weight_with_variations('qgl_weight', qgl_wgt, up=qgl_wgt*qgl_wgt, down=np.ones(numevents, dtype=float))
            
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
            genJetMass_filtered = genJetMass          
            gjets = df.GenJet     
            gleptons = df.GenPart[(df.GenPart.pdgId == 13) | (df.GenPart.pdgId == 11) | (df.GenPart.pdgId == 15)]
            gleptons = gleptons
            
            gl_pair = gjets.cross(gleptons, nested=True)
            _,_,dr_gl = delta_r(gl_pair.i0.eta, gl_pair.i1.eta, gl_pair.i0.phi, gl_pair.i1.phi)
            isolated = (dr_gl > 0.3).all()
            gjets = gjets[isolated]
            has_two_jets = gjets.counts>1
            gjet1 = gjets[has_two_jets,0]
            gjet2 = gjets[has_two_jets,1]
            _,_,_, genJetMass_filtered[has_two_jets] = p4_sum(gjet1,gjet2)
            
        #---------------------------------------------------------------#        
        # Apply remaining cuts
        #---------------------------------------------------------------#        
        
        leading_jet_pt = np.zeros(numevents, dtype=bool)
        leading_jet_pt[jets.counts>0] = (jets[jets.counts>0][:,0][[pt_name]]>35.)
        
        vbf_cut = (dijet_variables['jj_mass']>400)&(dijet_variables['jj_dEta']>2.5)&leading_jet_pt
        
        #---------------------------------------------------------------#        
        # Calculate btag SF and apply btag veto
        #---------------------------------------------------------------#
        bjet_sel_mask = mask&two_jets&vbf_cut
        
        # Btag weight
        btag_wgt = np.ones(numevents)
        if is_mc:
            btag_wgt = btag_weights(self.btag_lookup, jets, pt_name, weights, bjet_sel_mask, numevents)
            weights.add_weight('btag_weight', btag_wgt)

        # Separate from ttH and VH phase space        
        nBtagLoose = jets[(jets.btagDeepB>self.parameters["btag_loose_wp"]) & (abs(jets.eta)<2.5)].counts
        nBtagMedium = jets[(jets.btagDeepB>self.parameters["btag_medium_wp"])  & (abs(jets.eta)<2.5)].counts
        mask = mask & (nBtagLoose<2) & (nBtagMedium<1)

        mass = (muon_variables['dimuon_mass']>115) & (muon_variables['dimuon_mass']<135)
 #       weights.effect_on_normalization(mask&two_jets&vbf_cut&mass)
        
        if self.debug:
            print("VBF category (unweighted):", sum(mask&two_jets&vbf_cut))
            print("VBF category (weighted): ", weights.df['nominal'][mask&two_jets&vbf_cut].sum())

        #---------------------------------------------------------------#        
        # Define categories
        #---------------------------------------------------------------#        
        
        category = np.empty(numevents, dtype=object)
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
                'btag_wgt': btag_wgt,
                'event': df.event.flatten(),
            })


        return {
                'variable_map': ret, 
                'category': category,
                'two_jets': two_jets, 
                'weights': weights, 
               }

        
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
