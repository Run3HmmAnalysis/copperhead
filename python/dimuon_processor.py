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
import copy

from python.utils import p4_sum, p4_sum_alt, delta_r, rapidity, cs_variables
from python.timer import Timer
from python.samples_info import SamplesInfo
from python.weights import Weights
from python.corrections import musf_lookup, musf_evaluator, pu_lookup, pu_evaluator, NNLOPS_Evaluator, roccor_evaluator
from python.corrections import qgl_weights, puid_weights, btag_weights, geofit_evaluator, fsr_evaluator
from python.stxs_uncert import vbf_uncert_stage_1_1, stxs_lookups
from python.mass_resolution import mass_resolution_purdue, mass_resolution_pisa

import gc
    
def get_regions(mass):
    regions = {
        "z-peak": ((mass>76) & (mass<106)),
        "h-sidebands": ((mass>110) & (mass<115.03)) | ((mass>135.03) & (mass<150)),
        "h-peak": ((mass>115.03) & (mass<135.03)),
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
    def __init__(self, samp_info,\
                 do_timer=False, save_unbin=True, do_lheweights=False,\
                 do_jecunc=False, do_jerunc=False, do_pdf=True, auto_pu=True, debug=False, pt_variations=['nominal']): 
        from config.parameters import parameters
        from config.variables import variables
        if not samp_info:
            print("Samples info missing!")
            return
        self.auto_pu = auto_pu
        self.samp_info = samp_info
        self.year = self.samp_info.year
        self.debug = debug
        self.mass_window = [76, 150]
        self.save_unbin = save_unbin
        self.pt_variations = pt_variations
        self.do_pdf = do_pdf
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
        weights_ = ['nominal', 'lumi', 'genwgt', 'nnlops', 'btag_wgt']
        #variated_weights = ['pu_wgt', 'muSF', 'l1prefiring_wgt', 'qgl_wgt', 'LHEFac', 'LHERen']
        variated_weights = ['pu_wgt', 'muID', 'muIso', 'muTrig', 'l1prefiring_wgt', 'qgl_wgt', 'LHEFac', 'LHERen', 'pdf_2rms']
        self.sths_names = ["Yield","PTH200","Mjj60","Mjj120","Mjj350","Mjj700","Mjj1000","Mjj1500","PTH25","JET01"]
        variated_weights.extend(["THU_VBF_"+name for name in self.sths_names])
        self.btag_systs = ["jes", "lf", "hfstats1", "hfstats2","cferr1", "cferr2","hf", "lfstats1", "lfstats2"]
        variated_weights.extend(["btag_wgt_"+name for name in self.btag_systs])
        
        for wgt in weights_:
            if 'nominal' in wgt:
                variables.append(Variable("wgt_nominal", "wgt_nominal", 1, 0, 1))
            else:
                variables.append(Variable(f"wgt_{wgt}_off", f"wgt_{wgt}_off", 1, 0, 1))

        for wgt in variated_weights:
            variables.append(Variable(f"wgt_{wgt}_up", f"wgt_{wgt}_up", 1, 0, 1))
            variables.append(Variable(f"wgt_{wgt}_down", f"wgt_{wgt}_down", 1, 0, 1))
            variables.append(Variable(f"wgt_{wgt}_off", f"wgt_{wgt}_off", 1, 0, 1))

        if ('2016' in self.year) and self.do_pdf:
            for i in range(100):
                variables.append(Variable(f"pdf_mcreplica{i}", f"pdf_mcreplica{i}", 1, 0, 1))
            
#        if self.evaluate_dnn:
#            variables.append(Variable(f"dnn_score_nominal", f"dnn_score_nominal", 12, 0, self.parameters["dnn_max"]))
#            if self.do_jecunc:
#                for v_name in self.parameters["jec_unc_to_consider"]:
#                    variables.append(Variable(f"dnn_score_{v_name}_up", f"dnn_score_{v_name}_up", 12, 0,\
#                                              self.parameters["dnn_max"]))
#                    variables.append(Variable(f"dnn_score_{v_name}_down", f"dnn_score_{v_name}_down", 12, 0,\
#                                              self.parameters["dnn_max"]))
        
        self.vars_unbin = set([v.name for v in variables])
        
        dataset_axis = hist.Cat("dataset", "")
        region_axis = hist.Cat("region", "") # Z-peak, Higgs SB, Higgs peak
        channel_axis = hist.Cat("channel", "") # ggh or VBF  
        syst_axis = hist.Cat("syst", "")
        
        acc_dicts = {}
        if self.save_unbin:
            for jet_pt_var in self.pt_variations:
#                if jet_pt_var not in self.pt_variations: continue
                unbin_dict = {}
                for varname in self.vars_unbin:
                    for c in self.channels:
                        for r in self.regions:
                            unbin_dict[f'{varname}_{c}_{r}'] = processor.column_accumulator(np.ndarray([]))
                            # have to encode everything into the name because having multiple axes isn't possible
                acc_dicts[jet_pt_var] = processor.dict_accumulator(unbin_dict)

        ### --------------------------------------- ###
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
        self.stxs_acc_lookups, self.powheg_xsec_lookup = stxs_lookups()
        
        ### Prepare evaluator for corrections that can be loaded together ###        
        zpt_filename = self.parameters['zpt_weights_file']
        puid_filename = self.parameters['puid_sf_file']
        rescalib_files = []
        
        self.extractor = extractor()
        self.extractor.add_weight_sets([f"* * {zpt_filename}"])
        self.extractor.add_weight_sets([f"* * {puid_filename}"])
        self.extractor.add_weight_sets([f"* * data/mass_res_pisa/muonresolution.root"])
        
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
        self.Jet_transformer_JER = JetTransformer(jec=None, jer = JER, jersf = JERsf)
        self.Jet_transformer = JetTransformer(jec=JECcorrector,junc=JECuncertainties)
        self.JECcorrector_Data = {}
        self.Jet_transformer_data = {}
        self.data_runs = list(self.parameters['jec_unc_names_data'].keys())
        for run in self.data_runs:
            self.JECcorrector_Data[run] = FactorizedJetCorrector(**{name: Jetevaluator[name] for name in\
                                                          self.parameters['jec_names_data'][run]})
            JECuncertainties_Data = JetCorrectionUncertainty(**{name:Jetevaluator[name] for name in\
                                                                self.parameters['junc_names_data'][run]})
            self.Jet_transformer_data[run] = JetTransformer(jec=self.JECcorrector_Data[run],junc=JECuncertainties_Data)
            
        all_jec_names = [name for name in dir(Jetevaluator) if self.parameters['jec_unc_sources'] in name]
        self.JECuncertaintySources = JetCorrectionUncertainty(**{name: Jetevaluator[name] for name in all_jec_names})
        self.jet_unc_names = list(self.JECuncertaintySources.levels)
        
        self.do_jecunc = False
        self.do_jerunc = False
        for ptvar in self.pt_variations:
            if ptvar.replace('_up','').replace('_down','') in self.parameters["jec_unc_to_consider"]:
                self.do_jecunc = True
            if ptvar.replace('_up','').replace('_down','') in ['jer1','jer2','jer3','jer4','jer5','jer6']:
                self.do_jerunc = True


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
        self.debug=False
        
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

        if is_mc:
            pu_distribution = df.Pileup.nTrueInt
        df = df[hlt&(df.Muon.counts>1)]
    
        if self.debug:
            print("Passing HLT and at least 2 muons (any): ", len(df))
            
            
        #---------------------------------------------------------------#  
        # From now on, number of events will remain unchanged (size of 'mask').
        # Variable 'mask' will be used to store global event selection        
        # Apply HLT, lumimask, genweights, PU weights and L1 prefiring weights
        #---------------------------------------------------------------#            
        
        numevents = df.shape[0]
        weights = Weights(df)
        
        nTrueInt = df.Pileup.nTrueInt.flatten() if is_mc else np.zeros(numevents, dtype=bool)
        event_variables = {
                'run': df.run.flatten(),
                'event': df.event.flatten(),
                'npv': df.PV.npvs.flatten(),
                'nTrueInt': nTrueInt,
            }

        if is_mc:    
            mask = np.ones(numevents, dtype=bool)
            
            #---------------------------------------------------------------#        
            # Apply gen.weights, pileup weights, lumi weights, L1 prefiring weights
            # 
            #---------------------------------------------------------------# 
            
            genweight = df.genWeight.flatten()
            weights.add_weight('genwgt', genweight)    
            if self.auto_pu:
                self.pu_lookup = pu_lookup(self.parameters, 'nom', auto=pu_distribution)
                self.pu_lookup_up = pu_lookup(self.parameters, 'up', auto=pu_distribution)
                self.pu_lookup_down = pu_lookup(self.parameters, 'down', auto=pu_distribution)
            pu_weight = pu_evaluator(self.pu_lookup, numevents, df.Pileup.nTrueInt)
            pu_weight_up = pu_evaluator(self.pu_lookup_up, numevents, df.Pileup.nTrueInt)
            pu_weight_down = pu_evaluator(self.pu_lookup_down, numevents, df.Pileup.nTrueInt)
            weights.add_weight_with_variations('pu_wgt', pu_weight, pu_weight_up, pu_weight_down)

            
            if dataset in self.lumi_weights:
                weights.add_weight('lumi', self.lumi_weights[dataset])
             
            
            if self.parameters["do_l1prefiring_wgts"]:
                weights.add_weight_with_variations('l1prefiring_wgt',\
                                                   df.L1PreFiringWeight.Nom.flatten(),\
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
        muons_phi_raw = muons_phi
        muons_mass = mu.mass.flatten()
        muons_iso = mu.pfRelIso04_all.flatten() 
        muons_iso_raw = muons_iso
        has_fsr = np.zeros(len(mu.pt.flatten()), dtype=bool)
        
        roch_corr, roch_err = roccor_evaluator(self.roccor_lookup, is_mc, mu)
        
        muons_pt_roch = muons_pt*roch_corr
#        muons_pt_scale_up = muons_pt+muons_pt*roch_err
#        muons_pt_scale_down = muons_pt-muons_pt*roch_err
        muons_pt = muons_pt_roch # Rochester should be still applied
        muons_pts = {'nominal':muons_pt}#, 'scale_up':muons_pt_scale_up, 'scale_down':muons_pt_scale_down   ,}
        if True: # reserved for loop over muon pT variations
#        for
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
                                      np.array(muons_pt, dtype=float), np.array(muons_pt_roch, dtype=float),\
                                      np.array(muons_eta, dtype=float), np.array(muons_phi, dtype=float),\
                                      np.array(muons_mass, dtype=float), np.array(muons_iso, dtype=float),\
                                      np.array(muons_fsrIndex, dtype=int), fsr_pt, fsr_eta, fsr_phi, fsr_iso, fsr_drEt2,\
                          has_fsr) 
            muons_pt_fsr = muons_pt
            
            # GeoFit correction
            if 'dxybs' in mu.columns:
                muons_dxybs = mu.dxybs.flatten()
                muons_charge = mu.charge.flatten()
                muons_pt = geofit_evaluator(muons_pt, muons_eta, muons_dxybs, muons_charge, self.year, ~has_fsr).flatten() 
            
            updated_attrs = {'pt': muons_pt, 'pt_raw': muons_pt_raw, 'pt_fsr': muons_pt_fsr,'pt_roch': muons_pt_roch,\
                             'eta':muons_eta, 'eta_raw':muons_eta_raw, 'phi':muons_phi, 'phi_raw':muons_phi_raw,\
                             'mass':muons_mass, 'pfRelIso04_all':muons_iso, 'pfRelIso04_all_raw':muons_iso_raw}
            muonarrays = {key:mu[key].flatten() for key in mu.columns}
            muonarrays.update(updated_attrs)
            self.muons_all = JaggedCandidateArray.candidatesfromcounts(mu.counts, **muonarrays)
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
                         & pass_muon_flags
                         ]    
        
            self.muons_all = self.muons_all[(self.muons_all.pt_fsr > self.parameters["muon_pt_cut"]) &\
                        (self.muons_all.pfRelIso04_all < self.parameters["muon_iso_cut"]) &\
                        self.muons_all[self.parameters["muon_id"]] 
                         ]
        
            two_os_muons = ((muons.counts == 2) & (muons['charge'].prod() == -1))
        
            electrons = df.Electron[(df.Electron.pt > self.parameters["electron_pt_cut"]) &\
                                     (abs(df.Electron.eta) < self.parameters["electron_eta_cut"]) &\
                                     (df.Electron[self.parameters["electron_id"]] == 1)]
                
            electron_veto = (electrons.counts==0)
            good_pv = (df.PV.npvsGood > 0)
        
            event_filter = (pass_event_flags & two_os_muons & electron_veto & good_pv).flatten()
        
            if self.debug:
                print("Has 2 OS muons passing selections, good PV and no electrons:", sum(event_filter))

            mask = mask & event_filter

            if self.timer:
                self.timer.add_checkpoint("Applied muon corrections")

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
            trig_muons = trig_muons[(trig_muons.id == 13) | (trig_muons.id == -13)]
            trigmuarrays.clear()
        
            # Muons that pass tight id and iso as well as leading muon pT cut     
            mu_for_trigmatch = muons[(muons.pt_raw > self.parameters["muon_leading_pt"]) #&\
                                  # (muons.pfRelIso04_all < self.parameters["muon_trigmatch_iso"]) &\
                                  # muons[self.parameters["muon_trigmatch_id"]]
                                    ]
        
            # For every such muon check if there is a L3 object within dR<0.1
            muTrig = mu_for_trigmatch.cross(trig_muons, nested = True)
            _,_,dr = delta_r(muTrig.i0.eta_raw, muTrig.i1.eta, muTrig.i0.phi, muTrig.i1.phi)
            has_matched_trigmuon = (dr < self.parameters["muon_trigmatch_dr"]).any()
        
            # Events where there is a trigger object matched to a tight-ID tight-Iso muon passing leading pT cut
            event_passing_trig_match = (mu_for_trigmatch[has_matched_trigmuon].counts>0).flatten()
        
            mask = mask & pass_leading_pt #& event_passing_trig_match
        
            if self.debug:
                print("Leading pT cut, trigger matching:", sum(mask))
                
            if self.timer:
                self.timer.add_checkpoint("Applied trigger matching")

        
            #---------------------------------------------------------------#        
            # Initialize and fill dimuon and muon variables
            #---------------------------------------------------------------# 
        
            two_muons = muons.counts==2
        
            dimuon_variable_names = ['dimuon_mass', 'dimuon_mass_res', 'dimuon_mass_res_rel', 'dimuon_ebe_mass_res',\
                                     'dimuon_ebe_mass_res_rel', 'dimuon_pt', 'dimuon_eta', 'dimuon_phi', 'dimuon_dEta',\
                                     'dimuon_dPhi', 'dimuon_dR', 'dimuon_rap',\
                                     'dimuon_cos_theta_cs','dimuon_phi_cs']
            dimuon_variables = {}
            for n in dimuon_variable_names:
                dimuon_variables[n] = np.zeros(numevents)

            dimuon_variables['dimuon_pt'][two_muons],\
            dimuon_variables['dimuon_eta'][two_muons],\
            dimuon_variables['dimuon_phi'][two_muons],\
            dimuon_variables['dimuon_mass'][two_muons],\
            dimuon_variables['dimuon_rap'][two_muons] = p4_sum(mu1[two_muons], mu2[two_muons])

            dimuon_variables['dimuon_dEta'][two_muons],\
            dimuon_variables['dimuon_dPhi'][two_muons],\
            dimuon_variables['dimuon_dR'][two_muons] = delta_r(mu1[two_muons].eta.flatten(),\
                                                                                 mu2[two_muons].eta.flatten(),\
                                                                                 mu1[two_muons].phi.flatten(),\
                                                                                 mu2[two_muons].phi.flatten())

            dimuon_variables['dimuon_ebe_mass_res'][two_muons] = mass_resolution_purdue(is_mc,self.evaluator,mu1,mu2,\
                                                                                    dimuon_variables['dimuon_mass'],\
                                                                                    two_muons, self.year)
            dimuon_variables['dimuon_ebe_mass_res_rel'][two_muons] = (dimuon_variables['dimuon_mass_res'][two_muons] /\
                                                              dimuon_variables['dimuon_mass'][two_muons]).flatten()


            dimuon_variables['dimuon_mass_res_rel'][two_muons] = mass_resolution_pisa(self.extractor, mu1, mu2, two_muons)
            dimuon_variables['dimuon_mass_res'][two_muons] = dimuon_variables['dimuon_mass_res_rel'][two_muons]*\
                                                                dimuon_variables['dimuon_mass'][two_muons]
            
            dimuon_variables['dimuon_cos_theta_cs'][two_muons],\
            dimuon_variables['dimuon_phi_cs'][two_muons] = cs_variables(mu1,mu2,two_muons)

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
        
            if self.timer:
                self.timer.add_checkpoint("Filled muon variables")

        
        #---------------------------------------------------------------#        
        # Prepare jets
        #---------------------------------------------------------------# 
        jetarrays = {key:df.Jet[key].flatten() for key in df.Jet.columns} 
        jetarrays.update(**{'ptRaw':(df.Jet.pt * (1-df.Jet.rawFactor)).flatten(),\
                            'massRaw':(df.Jet.mass * (1-df.Jet.rawFactor)).flatten(),\
                            'rho': (df.Jet.pt.ones_like()*df.fixedGridRhoFastjetAll).flatten() })
#        jet_has_matched_muon = ((df.Jet.matched_muons.pt>15)&(df.Jet.matched_muons.mediumId)&(df.Jet.matched_muons.pfRelIso04_all<0.25)).any()
        jet_has_matched_muon = (df.Jet.matched_muons.pt>0).any()
        jwmm = df.Jet[jet_has_matched_muon].flatten()
        mm = jwmm.matched_muons[:,0]

        _,_,dr_mujet = delta_r(jwmm.eta, mm.eta, jwmm.phi, mm.phi)
        jet_matched_muon_dr = np.full(len(df.Jet.pt.flatten()), -999.)
        jet_matched_muon_pt = np.full(len(df.Jet.pt.flatten()), -999.)
        jet_matched_muon_iso = np.full(len(df.Jet.pt.flatten()), -999.)
        jet_matched_muon_id = np.full(len(df.Jet.pt.flatten()), -999.)

        jet_matched_muon_dr[jet_has_matched_muon.flatten()] = dr_mujet
        jet_matched_muon_pt[jet_has_matched_muon.flatten()] = mm.pt
        jet_matched_muon_iso[jet_has_matched_muon.flatten()] = mm.pfRelIso04_all
        jet_matched_muon_id[jet_has_matched_muon.flatten()] = mm.mediumId

        jetarrays.update(**{'has_matched_muon': np.array(jet_has_matched_muon.flatten(), dtype=int), 
                            'matched_muon_dr':jet_matched_muon_dr.flatten(),
                            'matched_muon_pt':jet_matched_muon_pt.flatten(),
                            'matched_muon_iso':jet_matched_muon_iso.flatten(),
                            'matched_muon_id':jet_matched_muon_id.flatten(),
                        })
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
        if self.timer:
            self.timer.add_checkpoint("Prepared jets")
 
        #---------------------------------------------------------------#        
        # Apply JEC, get JEC variations
        #---------------------------------------------------------------#
    
        self.do_jec = False
        if ('data' in dataset) and ('2018' in self.year):
            self.do_jec = True
        jet_variation_names = ['nominal']
        
        if self.do_jec or self.do_jecunc: 
            cols = {'pt':'__fast_pt',
                    'eta':'__fast_eta',
                    'phi':'__fast_phi',
                    'mass':'__fast_mass',
                    }
            cols.update(**{k:k for k in ['qgl','btagDeepB','ptRaw','massRaw','rho',\
                                         'area','jetId','puId','matched_muons',\
                                         'has_matched_muon', 'matched_muon_dr',\
                                         'matched_muon_pt', 'matched_muon_iso', 'matched_muon_id'
                                     ]})
            if self.year=="2017":
                cols.update(**{'puId17':'puId17'})
            if is_mc:
                cols.update({k:k for k in ['genJetIdx','ptGenJet','hadronFlavour', 'partonFlavour']})
            jetarrays = {key:jets[v].flatten() for key, v in cols.items()}
            jets = JaggedCandidateArray.candidatesfromcounts(jets.counts, **jetarrays)
            jetarrays.clear()
                        
            if is_mc:
                if self.do_jecunc:
                    for junc_name in self.jet_unc_names:
                        if junc_name not in self.parameters["jec_unc_to_consider"]: continue
                        if (f"{junc_name}_up" not in self.pt_variations) and (f"{junc_name}_down" not in self.pt_variations):
                            continue
                        jec_up_down = get_jec_unc(junc_name, jets.pt, jets.eta, self.JECuncertaintySources)
                        jec_corr_up, jec_corr_down = jec_up_down[:,:,0], jec_up_down[:,:,1]
                        pt_name_up = f"pt_{junc_name}_up"
                        pt_name_down = f"pt_{junc_name}_down"
                        jets.add_attributes(**{pt_name_up: jets.pt*jec_corr_up, pt_name_down: jets.pt*jec_corr_down})
                        if (f"{junc_name}_up" in self.pt_variations):
                            jet_variation_names += [f"{junc_name}_up"]
                        if (f"{junc_name}_down" in self.pt_variations):
                            jet_variation_names += [f"{junc_name}_down"]
                    
            else:
                if self.do_jec:
                    for run in self.data_runs: # 'A', 'B', 'C', 'D', etc...
                        if run in dataset: # dataset name is something like 'data_B'
                            self.Jet_transformer_data[run].transform(jets, forceStochastic=False) 

        if is_mc and self.do_jerunc:
            jet_pt_jec = jets.pt
            self.Jet_transformer_JER.transform(jets, forceStochastic=False)
            jet_pt_jec_jer = jets.pt
            jer_sf = (jet_pt_jec_jer-jets.ptGenJet) / \
                    (jet_pt_jec-jets.ptGenJet+(jet_pt_jec==jets.ptGenJet)*(jet_pt_jec_jer-jet_pt_jec))
            jer_down_sf = (jets.pt_jer_down-jets.ptGenJet)/(jet_pt_jec-jets.ptGenJet+(jet_pt_jec==jets.ptGenJet)*10.)
            jet_pt_jer_down = jets.ptGenJet + (jet_pt_jec - jets.ptGenJet)*(jer_down_sf/jer_sf)
            jer_categories = {
                'jer1' : (abs(jets.eta)<1.93),
                'jer2' : (abs(jets.eta)>1.93)&(abs(jets.eta)<2.5),
                'jer3' : (abs(jets.eta)>2.5)&(abs(jets.eta)<3.139)&(jets.pt<50),
                'jer4' : (abs(jets.eta)>2.5)&(abs(jets.eta)<3.139)&(jets.pt>50),
                'jer5' : (abs(jets.eta)>3.139)&(jets.pt<50),
                'jer6' : (abs(jets.eta)>3.139)&(jets.pt>50),
            }
            for jer_unc_name, jer_cut in jer_categories.items():
                if (f"{jer_unc_name}_up" not in self.pt_variations) and (f"{jer_unc_name}_down" not in self.pt_variations):
                    continue
                pt_name_up = f"pt_{jer_unc_name}_up"
                pt_name_down = f"pt_{jer_unc_name}_down"
                jet_pt_up = copy.deepcopy(jet_pt_jec)
                jet_pt_down = copy.deepcopy(jet_pt_jec)
                jet_pt_up[jer_cut] = jet_pt_jec_jer[jer_cut]
                jet_pt_down[jer_cut] = jet_pt_jer_down[jer_cut]
                jets.add_attributes(**{pt_name_up: jet_pt_up, pt_name_down: jet_pt_down})
                if (f"{jer_unc_name}_up" in self.pt_variations):
                    jet_variation_names += [f"{jer_unc_name}_up"]
                if (f"{jer_unc_name}_down" in self.pt_variations):
                    jet_variation_names += [f"{jer_unc_name}_down"]
                            
        jets.add_attributes(**{'pt_nominal': jets.pt})
        jets = jets[jets.pt.argsort()]  

        if self.timer:
            self.timer.add_checkpoint("Applied JEC (if enabled)")

        
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
                nnlopsw_new = np.ones(numevents, dtype=float)
                gen_njets = df.GenJet[df.GenJet.pt>30.].counts
                gen_higgs = df.GenPart[(df.GenPart.pdgId == 25)&(df.GenPart.status == 62)]
                has_higgs = gen_higgs.counts>0
                gen_hpt = np.zeros(numevents, dtype=float)
                gen_hpt[has_higgs] = gen_higgs[has_higgs].pt[:,0]
                if 'amc' in dataset:
#                    nnlopsw[has_higgs] = nnlops.evaluate(gen_hpt[has_higgs], gen_njets[has_higgs], "mcatnlo")
                    nnlopsw_new = nnlops.evaluate(df.HTXS.Higgs_pt, df.HTXS.njets30, "mcatnlo")
                elif 'powheg' in dataset:
#                    nnlopsw[has_higgs] = nnlops.evaluate(gen_hpt[has_higgs], gen_njets[has_higgs], "powheg")
                    nnlopsw_new = nnlops.evaluate(df.HTXS.Higgs_pt, df.HTXS.njets30, "powheg")
                weights.add_weight('nnlops', nnlopsw_new)

#            if 'dy' in dataset:
#                zpt_weight = np.ones(numevents, dtype=float)
#                zpt_weight[two_muons] = self.evaluator[self.zpt_path](dimuon_variables['dimuon_pt'][two_muons]).flatten()
#                weights.add_weight('zpt_wgt', zpt_weight)

            muID, muID_up, muID_down,\
            muIso, muIso_up, muIso_down,\
            muTrig, muTrig_up, muTrig_down = musf_evaluator(self.musf_lookup, self.year, numevents, muons)
            weights.add_weight_with_variations('muID', muID, muID_up, muID_down)
            weights.add_weight_with_variations('muIso', muIso, muIso_up, muIso_down)           
#            weights.add_weight_with_variations('muTrig', muTrig, muTrig_up, muTrig_down)

            if self.do_lheweights and ('nominal' in self.pt_variations):
                lhefactor = 2. if ('dy_m105_160_amc' in dataset) and (('2017' in self.year) or ('2018' in self.year)) else 1.
                nLHEScaleWeight = df.LHEScaleWeight.counts

                lhe_ren_up = np.full(numevents, df.LHEScaleWeight[:,6]*lhefactor, dtype=float)
                lhe_ren_up[nLHEScaleWeight>8] = df.LHEScaleWeight[nLHEScaleWeight>8][:,7]*lhefactor
                lhe_ren_up[nLHEScaleWeight>30] = df.LHEScaleWeight[nLHEScaleWeight>30][:,34]*lhefactor
                lhe_ren_down = np.full(numevents, df.LHEScaleWeight[:,1]*lhefactor, dtype=float)
                lhe_ren_down[nLHEScaleWeight>8] = df.LHEScaleWeight[nLHEScaleWeight>8][:,1]*lhefactor
                lhe_ren_down[nLHEScaleWeight>30] = df.LHEScaleWeight[nLHEScaleWeight>30][:,5]*lhefactor  
                weights.add_only_variations('LHERen', lhe_ren_up, lhe_ren_down)
                lhe_fac_up = np.full(numevents, df.LHEScaleWeight[:,4]*lhefactor, dtype=float)
                lhe_fac_up[nLHEScaleWeight>8] = df.LHEScaleWeight[nLHEScaleWeight>8][:,5]*lhefactor
                lhe_fac_up[nLHEScaleWeight>30] = df.LHEScaleWeight[nLHEScaleWeight>30][:,24]*lhefactor
                lhe_fac_down = np.full(numevents, df.LHEScaleWeight[:,3]*lhefactor, dtype=float)
                lhe_fac_down[nLHEScaleWeight>8] = df.LHEScaleWeight[nLHEScaleWeight>8][:,3]*lhefactor
                lhe_fac_down[nLHEScaleWeight>30] = df.LHEScaleWeight[nLHEScaleWeight>30][:,15]*lhefactor
                weights.add_only_variations('LHEFac', lhe_fac_up, lhe_fac_down)

            if ('vbf' in dataset) and ('dy' not in dataset) and ('nominal' in self.pt_variations):
                for i, name in enumerate(self.sths_names):
                    wgt_up = vbf_uncert_stage_1_1(i, df.HTXS.stage1_1_fine_cat_pTjet30GeV, 1.,\
                                                  self.stxs_acc_lookups, self.powheg_xsec_lookup)
                    wgt_down = vbf_uncert_stage_1_1(i, df.HTXS.stage1_1_fine_cat_pTjet30GeV, -1.,\
                                                    self.stxs_acc_lookups, self.powheg_xsec_lookup)
                    weights.add_only_variations("THU_VBF_"+name, wgt_up, wgt_down)
                    
        if self.timer:
            self.timer.add_checkpoint("Computed event weights")

        #---------------------------------------------------------------#
        # Calculate getJetMass
        #---------------------------------------------------------------#

        genJetMass = np.zeros(numevents, dtype=float)
        if is_mc:
            gjets = df.GenJet
            gleptons = df.GenPart[(df.GenPart.pdgId == 13) | (df.GenPart.pdgId == 11) | (df.GenPart.pdgId == 15) |\
                                 (df.GenPart.pdgId == -13) | (df.GenPart.pdgId == -11) | (df.GenPart.pdgId == -15)]
            gl_pair = gjets.cross(gleptons, nested=True)
            _,_,dr_gl = delta_r(gl_pair.i0.eta, gl_pair.i1.eta, gl_pair.i0.phi, gl_pair.i1.phi)
            isolated = (dr_gl > 0.3).all()
            gjets = gjets[isolated]
            has_two_jets = gjets.counts>1
            gjet1 = gjets[has_two_jets,0]
            gjet2 = gjets[has_two_jets,1]
            _,_,_, genJetMass[has_two_jets],_ = p4_sum(gjet1,gjet2)

        #---------------------------------------------------------------#        
        # Loop over JEC variations and fill jet variables
        #---------------------------------------------------------------#        

        ret_jec_loop = {}
        variable_map = {}
        for v_name in jet_variation_names:
            ret_jec_loop[v_name] = self.jec_loop(v_name, is_mc, df, dataset, mask, muons, mu1, mu2, muon_variables,\
                                                 event_variables, two_muons, jets, weights, numevents, genJetMass)
            
        if self.timer:
            self.timer.add_checkpoint("Completed JEC loop")
            
        variable_map = ret_jec_loop['nominal']['variable_map']
        weights = ret_jec_loop['nominal']['weights']
        two_jets = ret_jec_loop['nominal']['two_jets']
        category = ret_jec_loop['nominal']['category']

        weights.effect_on_normalization((category=='vbf'))

#        print(df.event[(category=='vbf')&(ret_jec_loop['nominal']['variable_map']['dimuon_mass']>115)&(ret_jec_loop['nominal']['variable_map']['dimuon_mass']<135)])
        evnum = 425742024
        try:
#            print(weights.wgts.loc[evnum,:])
#            print(weights.wgts.mean())
            for vname, var in variable_map.items():
                print(vname, var[df.event==evnum])
        except:
            pass

        #---------------------------------------------------------------#        
        # PDF variations
        #---------------------------------------------------------------#         

        if self.do_pdf and is_mc and ('nominal' in self.pt_variations):
            pdf_rms = np.zeros(numevents, dtype=float)
            if ("dy" in dataset or "ewk" in dataset or "ggh" in dataset or "vbf" in dataset) and ('mg' not in dataset):
                pdf_wgts = df.LHEPdfWeight[:,0:self.parameters["n_pdf_variations"]]
                if '2016' in self.year:
                    max_replicas = 0
                    if 'dy' in dataset: max_replicas = 100
                    elif 'ewk' in dataset: max_replicas = 33
                    else: max_replicas = 100
                    for i in range(max_replicas):
                        ret_jec_loop['nominal']['variable_map'][f"pdf_mcreplica{i}"] = pdf_wgts[:,i]
                else:
                    weights.add_only_variations("pdf_2rms", (1+2*pdf_wgts.std()), (1-2*pdf_wgts.std()))           
                    
        #---------------------------------------------------------------#        
        # Fill outputs
        #---------------------------------------------------------------#                        

        regions = get_regions(variable_map['dimuon_mass'])            

        if self.save_unbin:
            for jec_var in jet_variation_names:
                if jec_var not in self.pt_variations: continue
                var_map = ret_jec_loop[jec_var]['variable_map']
                categ = ret_jec_loop[jec_var]['category']
                weights = ret_jec_loop[jec_var]['weights']
                for wgt in weights.df.columns:
                    var_map[f'wgt_{wgt}'] = weights.get_weight(wgt)                
                for v in var_map:
                    if (v not in self.vars_unbin) and ('wgt_' not in v) and ('mcreplica' not in v): continue
                    for cname in self.channels:
                        ccut = (categ==cname)
                        for rname, rcut in regions.items():
                            if ('dy_m105_160_vbf_amc' in dataset) and ('vbf' in cname):
                                ccut = ccut & (genJetMass > 350.)
                            if ('dy_m105_160_amc' in dataset) and ('vbf' in cname):
                                ccut = ccut & (genJetMass <= 350.)
                            value = np.array(var_map[v][rcut & ccut]).ravel()
                            output[jec_var][f'{v}_{cname}_{rname}'] += processor.column_accumulator(value)

        if self.timer:
            self.timer.add_checkpoint("Filled outputs")
            
        if self.timer:
            self.timer.summary()

        variable_map.clear()
        for ret in ret_jec_loop.values():
            ret['variable_map'].clear()
            ret.clear()
        ret_jec_loop.clear()
        return output

    
    def jec_loop(self, variation, is_mc, df, dataset, mask, muons, mu1, mu2, muon_variables, event_variables, two_muons, jets, weights, numevents, genJetMass):
        weights = copy.deepcopy(weights)
        #---------------------------------------------------------------#        
        # Initialize jet-related variables
        #---------------------------------------------------------------#        
        
        jet1_variable_names = ['jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_qgl', 'jet1_id', 'jet1_puid', 'jet1_has_matched_muon', 'jet1_matched_muon_dr',
                               'jet1_matched_muon_pt', 'jet1_matched_muon_iso', 'jet1_matched_muon_id']
        jet1_variables = {}
        for n in jet1_variable_names:
            jet1_variables[n] = np.full(numevents, -999.)

        jet2_variable_names = ['jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_qgl', 'jet2_id', 'jet2_puid', 'jet2_has_matched_muon', 'jet2_matched_muon_dr',
                               'jet2_matched_muon_pt', 'jet2_matched_muon_iso', 'jet2_matched_muon_id']
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

        mmjj_variable_names = ['mmjj_pt', 'mmjj_eta', 'mmjj_phi', 'mmjj_mass', 'rpt', 'zeppenfeld','ll_zstar_log']
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

#        # Alternative way (doesn't take into account FSR)
#        match_mu = jets.matched_muons
#        deltar_mujet_ok = ((match_mu.pfRelIso04_all>0.25) | (~match_mu.mediumId) | (match_mu.pt<20)).all().flatten()

        mujet = jets.cross(self.muons_all, nested=True)
        _,_,deltar_mujet = delta_r(mujet.i0.eta, mujet.i1.eta_raw, mujet.i0.phi, mujet.i1.phi_raw)
        deltar_mujet_ok = (deltar_mujet > self.parameters["min_dr_mu_jet"]).all()

        jet_selection = ((jets[pt_name] > self.parameters["jet_pt_cut"]) &\
                         (abs(jets.eta) < self.parameters["jet_eta_cut"])) & deltar_mujet_ok
            
        #---------------------------------------------------------------#        
        # Calculate PUID scale factors and apply PUID
        #---------------------------------------------------------------#        
       
        # Jet PU ID
        jet_puid_opt = self.parameters["jet_puid"]
        puId = jets.puId17 if self.year=="2017" else jets.puId
        jet_puid_wps = {
            "loose": ( puId >=4 ) | ( jets[pt_name] > 50 ),
            "medium": ( puId >=6 ) | ( jets[pt_name] > 50 ),
            "tight": ( puId >=7 ) | ( jets[pt_name] > 50 ),
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
        #if is_mc:
            #puid_weight = puid_weights(self.evaluator, self.year, jets, pt_name, jet_puid_opt, jet_puid, numevents)
            #weights.add_weight('puid_wgt', puid_weight)

        jet_selection = jet_selection & jet_puid
        jets = jets[jet_selection]
        
        if self.timer:
            self.timer.add_checkpoint("Selected jets")

        
        #---------------------------------------------------------------#        
        # Fill jet-related variables
        #---------------------------------------------------------------#        
        one_jet = (jets.counts>0)
        two_jets = (jets.counts>1)
        cols = {'pt':pt_name,
                'eta':'__fast_eta',
                'phi':'__fast_phi',
                'mass':'__fast_mass'}
        cols.update({k:k for k in ['qgl','jetId','puId', 'btagDeepB', 'has_matched_muon', 'matched_muon_dr', 'matched_muon_pt', 'matched_muon_iso', 'matched_muon_id']})
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
            qgl_wgt[one_jet&~two_jets] = 1.
            qgl_wgt = qgl_wgt/qgl_wgt[mask&two_jets].mean()
            weights.add_weight_with_variations('qgl_wgt', qgl_wgt, up=qgl_wgt*qgl_wgt, down=np.ones(numevents, dtype=float))

            
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

        jet1_variables['jet1_has_matched_muon'][one_jet] = jet1.has_matched_muon.flatten()
        jet1_variables['jet1_matched_muon_dr'][one_jet] = jet1.matched_muon_dr.flatten()
        jet1_variables['jet1_matched_muon_pt'][one_jet] = jet1.matched_muon_pt.flatten()
        jet1_variables['jet1_matched_muon_iso'][one_jet] = jet1.matched_muon_iso.flatten()
        jet1_variables['jet1_matched_muon_id'][one_jet] = jet1.matched_muon_id.flatten()

        jet2_variables['jet2_has_matched_muon'][two_jets] = jet2.has_matched_muon.flatten()
        jet2_variables['jet2_matched_muon_dr'][two_jets] = jet2.matched_muon_dr.flatten()
        jet2_variables['jet2_matched_muon_pt'][two_jets] = jet2.matched_muon_pt.flatten()
        jet2_variables['jet2_matched_muon_iso'][two_jets] = jet2.matched_muon_iso.flatten()
        jet2_variables['jet2_matched_muon_id'][two_jets] = jet2.matched_muon_id.flatten()

        dijet_variables['jj_pt'][two_jets],\
        dijet_variables['jj_eta'][two_jets],\
        dijet_variables['jj_phi'][two_jets],\
        dijet_variables['jj_mass'][two_jets],_ = p4_sum(jet1[two_jets], jet2[two_jets])
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
        ll_ystar = np.full(numevents, -999.)
        ll_zstar = np.full(numevents, -999.)
        jet1_rap = rapidity(jet1)[two_muons&two_jets]
        jet2_rap = rapidity(jet2)[two_muons&two_jets]
        ll_ystar[two_muons&two_jets] = muon_variables['dimuon_rap'][two_muons&two_jets] - (jet1_rap+jet2_rap)/2
        ll_zstar[two_muons&two_jets] = abs(ll_ystar[two_muons&two_jets] /(jet1_rap-jet2_rap))
        mmjj_variables['ll_zstar_log'][two_muons&two_jets] = np.log(ll_zstar[two_muons&two_jets])

        
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
        
        mmj_variables['mmj_min_dPhi'] = np.where(mmj_variables['mmj1_dPhi'],\
                                                 mmj_variables['mmj2_dPhi'],\
                                                 (mmj_variables['mmj1_dPhi'] < mmj_variables['mmj2_dPhi']))

        if self.timer:
            self.timer.add_checkpoint("Filled jet variables")

        #---------------------------------------------------------------#        
        # Fill soft activity jet variables
        #---------------------------------------------------------------#        
        
        sjarrays = {key:df.SoftActivityJet[key].flatten() for key in df.SoftActivityJet.columns}   
        sjarrays.update({'mass':0})
        softjets = JaggedCandidateArray.candidatesfromcounts(df.SoftActivityJet.counts, **sjarrays)
        
        softjet_variables['nsoftjets2'],\
        softjet_variables['htsoft2'] = self.get_softjet_vars(df, softjets, 2, muons, mu1, mu2, jets, jet1, jet2, two_muons, one_jet, two_jets)
        
        softjet_variables['nsoftjets5'],\
        softjet_variables['htsoft5'] = self.get_softjet_vars(df, softjets, 5, muons, mu1, mu2, jets, jet1, jet2, two_muons, one_jet, two_jets)

        if self.timer:
            self.timer.add_checkpoint("Calculated SA variables")

        
        #---------------------------------------------------------------#        
        # Apply remaining cuts
        #---------------------------------------------------------------#        
        
        leading_jet_pt = np.zeros(numevents, dtype=bool)
        leading_jet_pt[jets.counts>0] = (jets.pt[jets.counts>0][:,0]>35.)
        vbf_cut = (dijet_variables['jj_mass']>400)&(dijet_variables['jj_dEta']>2.5)&leading_jet_pt
        
        #---------------------------------------------------------------#        
        # Calculate btag SF and apply btag veto
        #---------------------------------------------------------------#
        bjet_sel_mask = mask&two_jets&vbf_cut
        
        # Btag weight
        btag_wgt = np.ones(numevents)
        if is_mc:
#            self.btag_systs = []
            btag_wgt, btag_syst = btag_weights(self.btag_lookup, self.btag_systs, jets, weights, bjet_sel_mask, numevents)
            weights.add_weight('btag_wgt', btag_wgt)
            for name,bs in btag_syst.items():
                up = bs[0]
                down = bs[1]
                weights.add_only_variations(f'btag_wgt_{name}', up, down)

        # Separate from ttH and VH phase space        
        nBtagLoose = jets[(jets.btagDeepB>self.parameters["btag_loose_wp"]) & (abs(jets.eta)<2.5)].counts
        nBtagMedium = jets[(jets.btagDeepB>self.parameters["btag_medium_wp"])  & (abs(jets.eta)<2.5)].counts
        mask = mask & (nBtagLoose<2) & (nBtagMedium<1)

        mass = (muon_variables['dimuon_mass']>115) & (muon_variables['dimuon_mass']<135)
        
        if self.debug:
            print("VBF category (unweighted):", sum(mask&two_jets&vbf_cut))
            print("VBF category (weighted): ", weights.df['nominal'][mask&two_jets&vbf_cut].sum())

        
        if self.timer:
            self.timer.add_checkpoint("Applied b-jet SF")

        #---------------------------------------------------------------#        
        # Define categories
        #---------------------------------------------------------------#            
        category = np.empty(numevents, dtype=object)
        category[mask&(~two_jets)] = 'ggh_01j'
        category[mask&two_jets&(~vbf_cut)] = 'ggh_2j'
        category[mask&two_jets&vbf_cut] = 'vbf'
        if 'dy' in dataset:
            two_jets_matched = np.zeros(numevents, dtype=bool)
            two_jets_matched[two_jets] = (jet1.genJetIdx[two_jets]>0)&(jet1.genJetIdx[two_jets]>0)
            category[mask&two_jets&vbf_cut&(~two_jets_matched)] = 'vbf_01j'
            category[mask&two_jets&vbf_cut&two_jets_matched] = 'vbf_2j'

        #---------------------------------------------------------------#        
        # Fill outputs
        #---------------------------------------------------------------#        
            
        ret = {}
        ret.update(event_variables)
        ret.update(muon_variables)
        ret.update(dijet_variables)
        ret.update(jet1_variables)
        ret.update(jet2_variables)
        ret.update(dijet_variables)
        ret.update(mmjj_variables)
        ret.update(mmj_variables)
        ret.update(softjet_variables)
        ret.update(**{
                'njets': jets.counts.flatten(),
                'met': df.MET.pt.flatten(),
                'btag_wgt': btag_wgt,
            })

        return {
                'variable_map': copy.deepcopy(ret),
                'category': copy.deepcopy(category),
                'two_jets': copy.deepcopy(two_jets),
                'weights': weights, 
               }


    def get_softjet_vars(self, df, softjets, cutoff, muons, mu1, mu2, jets, jet1, jet2, two_muons, one_jet, two_jets):
        nsoftjets = df[f'SoftActivityJetNjets{cutoff}']
        htsoft = df[f'SoftActivityJetHT{cutoff}']
        mask = two_muons
        mask1j = two_muons&one_jet
        mask2j = two_muons&two_jets

        sj_j = softjets.cross(jets,nested=True)
        sj_mu = softjets.cross(muons,nested=True)
        
        _,_,dr_sj_j = delta_r(sj_j.i0.eta, sj_j.i1.eta, sj_j.i0.phi, sj_j.i1.phi)
        _,_,dr_sj_mu = delta_r(sj_mu.i0.eta, sj_mu.i1.eta, sj_mu.i0.phi, sj_mu.i1.phi)
        
        closest_jet = sj_j[(dr_sj_j==dr_sj_j.min())&(dr_sj_j<0.4)]
        closest_mu = sj_mu[(dr_sj_mu==dr_sj_mu.min())&(dr_sj_mu<0.4)]
        
        sj_j1 = closest_jet.cross(jet1,nested=True)
        sj_j2 = closest_jet.cross(jet2,nested=True)
        sj_mu1 = closest_mu.cross(mu1,nested=True)
        sj_mu2 = closest_mu.cross(mu2,nested=True)
       
        _,_,dr_sj_j1 = delta_r(sj_j1.i0.eta, sj_j1.i1.eta, sj_j1.i0.phi, sj_j1.i1.phi)
        _,_,dr_sj_j2 = delta_r(sj_j2.i0.eta, sj_j2.i1.eta, sj_j2.i0.phi, sj_j2.i1.phi)
        _,_,dr_sj_mu1 = delta_r(sj_mu1.i0.eta, sj_mu1.i1.eta, sj_mu1.i0.phi, sj_mu1.i1.phi)
        _,_,dr_sj_mu2 = delta_r(sj_mu2.i0.eta, sj_mu2.i1.eta, sj_mu2.i0.phi, sj_mu2.i1.phi)
        
        j1match = (dr_sj_j1<0.4).any().any()
        j2match = (dr_sj_j2<0.4).any().any()
        mumatch = ((dr_sj_mu1<0.4).any().any()|(dr_sj_mu2<0.4).any().any())
        
        eta1cut1 = (sj_j1.i0.eta>sj_j1.i1.eta).any().any()
        eta2cut1 = (sj_j2.i0.eta>sj_j2.i1.eta).any().any()
        outer = (eta1cut1[mask2j])&(eta2cut1[mask2j])
        eta1cut2 = (sj_j1.i0.eta<sj_j1.i1.eta).any().any()
        eta2cut2 = (sj_j2.i0.eta<sj_j2.i1.eta).any().any()
        inner = (eta1cut2[mask2j])&(eta2cut2[mask2j])
                
        nsoftjets[mask] = df[f'SoftActivityJetNjets{cutoff}'][mask]-\
                            (mumatch[mask] & (df.SoftActivityJet.pt>cutoff)[mask]).sum().sum()
        nsoftjets[mask1j] = df[f'SoftActivityJetNjets{cutoff}'][mask1j]-\
                            ((mumatch[mask1j]|j1match[mask1j]) & (df.SoftActivityJet.pt>cutoff)[mask1j]).sum()
        nsoftjets[mask2j] = df[f'SoftActivityJetNjets{cutoff}'][mask2j]-\
                        ((mumatch[mask2j]|j1match[mask2j]|j2match[mask2j])&(df.SoftActivityJet.pt>cutoff)[mask2j]).sum()
        saj_filter = (mumatch[mask2j]|j1match[mask2j]|j2match[mask2j]|outer|inner)
        footprintSAJ = df.SoftActivityJet[mask2j][saj_filter]
        if footprintSAJ.shape[0]>0:
            htsoft[mask2j] = df[f'SoftActivityJetHT{cutoff}'][mask2j]-(footprintSAJ.pt*(footprintSAJ.pt>cutoff)).sum()  
        return nsoftjets, htsoft
  
    def postprocess(self, accumulator):
        return accumulator