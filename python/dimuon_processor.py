from coffea import hist, util
from coffea.analysis_objects import JaggedCandidateArray, JaggedCandidateMethods
import coffea.processor as processor
from coffea.lookup_tools import extractor, dense_lookup, txt_converters, rochester_lookup
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty, JetTransformer, JetResolution, JetResolutionScaleFactor
from coffea.btag_tools import BTagScaleFactor
from coffea.lumi_tools import LumiMask

import awkward
import uproot
import numpy as np
import numba
import pandas as pd

from python.utils import apply_roccor, p4_sum, NNLOPS_Evaluator
from python.timer import Timer
from python.samples_info import SamplesInfo

def get_regions(mass):
    regions = {
        "z-peak": ((mass>70) & (mass<110)),
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

def get_jec_unc(name, jet_pt, jet_eta, jesunc):
    idx_func = jesunc.levels.index(name)
    jec_unc_func = jesunc._funcs[idx_func]
    function_signature = jesunc._funcs[idx_func].signature
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

def apply_geofit(muons_pt, muons_eta, muons_dxybs, muons_charge, year):
    pt_cor = np.zeros(len(muons_pt.flatten()), dtype=float)

    d0_BS_charge_full = np.multiply(muons_dxybs.flatten(),muons_charge.flatten()) 
    good_ds = np.abs(d0_BS_charge_full)<999999.
    d0_BS_charge = d0_BS_charge_full[good_ds]
    pt = muons_pt.flatten()[good_ds]
    eta = muons_eta.flatten()[good_ds]
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
        pt_cor[good_ds][cuts[eta_i]] = factors[year][eta_i]*d0_BS_charge[cuts[eta_i]]*pt[cuts[eta_i]]**2/10000.0
    
    return (pt - pt_cor)
    

# https://github.com/jpata/hepaccelerate-cms/blob/f5965648f8a7861cb9856d0b5dd34a53ed42c027/tests/hmm/hmumu_utils.py#L1396
@numba.njit(parallel=True)
def correct_muon_with_fsr(muons_offsets, fsr_offsets, muons_pt, muons_eta, muons_phi,\
                          muons_mass, muons_iso, muons_fsrIndex,
                            fsr_pt, fsr_eta, fsr_phi):    
    for iev in numba.prange(len(muons_offsets) - 1):
        #loop over muons in event
        mu_first = muons_offsets[iev]
        mu_last = muons_offsets[iev + 1]
        for imu in range(mu_first, mu_last):
            #relative FSR index in the event
            fsr_idx_relative = muons_fsrIndex[imu]

            if (fsr_idx_relative >= 0) and (muons_pt[imu]>20):
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
 
def get_variated_jet_variables(v_name, jets, dimuons, one_jet, two_jets, event_weight):
    if v_name=='nominal': 
        return {}
    
    cols = {
            f'pt_{v_name}': 'pt',
            '__fast_eta': 'eta',
            '__fast_phi': 'phi',
            '__fast_mass': 'mass',
            'qgl': 'qgl',
    }
    
    ret = {}

    jet1_mask = np.zeros(len(event_weight))
    jet1_mask[one_jet] = 1
    jet2_mask = np.zeros(len(event_weight))
    jet2_mask[two_jets] = 1
    
    my_jets_arrays = {v:jets[key].flatten() for key, v in cols.items()}
    my_jet1_arrays = {v:jets[one_jet][jets[one_jet].pt.argmax()][key].flatten() for key, v in cols.items()}
    my_jet2_arrays = {v:jets[two_jets][jets[two_jets].pt.argmin()][key].flatten() for key, v in cols.items()}
                
    jet1 = JaggedCandidateArray.candidatesfromcounts(jet1_mask, **my_jet1_arrays)
    jet2 = JaggedCandidateArray.candidatesfromcounts(jet2_mask, **my_jet2_arrays)
    jets = JaggedCandidateArray.candidatesfromcounts(jets.counts, **my_jets_arrays) 

    dijet_pairs = jets[two_jets, 0:2]
    dijet_mask = np.zeros(len(event_weight))
    dijet_mask[two_jets] = 2
    dijet_jca = JaggedCandidateArray.candidatesfromcounts(
        dijet_mask,
        pt=dijet_pairs.pt.flatten(),
        eta=dijet_pairs.eta.flatten(),
        phi=dijet_pairs.phi.flatten(),
        mass=dijet_pairs.mass.flatten(),
    )
        
    dijet = dijet_jca.distincts()
    dijet = dijet.p4.sum()

    dijet_deta = np.full(len(event_weight), -999.)
    dijet_deta[two_jets] = abs(jet1[two_jets].eta - jet2[two_jets].eta)
    
    dijet_dphi = np.full(len(event_weight), -999.)
    dijet_dphi[two_jets] = abs(jet1[two_jets].p4.delta_phi(jet2[two_jets].p4))

    zeppenfeld = np.full(len(event_weight), -999.)
    zeppenfeld[two_jets] = (dimuons.eta[two_jets] - 0.5*(jet1.eta[two_jets] + jet2.eta[two_jets]))
        
    rpt = np.full(len(event_weight), -999.)
    mmjj_pt = np.full(len(event_weight), 0.)
    mmjj_eta = np.full(len(event_weight), -999.)
    mmjj_phi = np.full(len(event_weight), -999.)
    mmjj_mass = np.full(len(event_weight), 0.)
    mmjj_pt[two_jets], mmjj_eta[two_jets], mmjj_phi[two_jets], mmjj_mass[two_jets] = p4_sum(dimuons[two_jets], dijet[two_jets])
    rpt[two_jets] =  mmjj_pt[two_jets]/(dimuons.pt[two_jets] + jet1.pt[two_jets] + jet2.pt[two_jets])


    ret["jet1_pt"] = jet1["__fast_pt"]
    ret["jet1_eta"] = jet1["__fast_eta"]
    ret["jet1_phi"] = jet1["__fast_phi"]
    ret["jet1_qgl"] = jet1["qgl"]  
    
    ret["jet2_pt"] = jet2["__fast_pt"]
    ret["jet2_eta"] = jet2["__fast_eta"]
    ret["jet2_phi"] = jet2["__fast_phi"]
    ret["jet2_qgl"] = jet2["qgl"] 
    
    ret["jj_deta"] = dijet_deta
    ret["jj_dphi"] = dijet_dphi
    ret["jj_mass"] = dijet.mass
    ret["jj_pt"] = dijet.pt
    ret["jj_eta"] = dijet.eta
    ret["jj_phi"] = dijet.phi
    ret["zeppenfeld"] = zeppenfeld
    ret["rpt"] = rpt
    ret["mmjj_pt"] = mmjj_pt
    ret["mmjj_eta"] = mmjj_eta
    ret["mmjj_phi"] = mmjj_phi
    ret["mmjj_mass"] = mmjj_mass
    
    
    return ret
    
# Look at ProcessorABC documentation to see the expected methods and what they are supposed to do
# https://coffeateam.github.io/coffea/api/coffea.processor.ProcessorABC.html
class DimuonProcessor(processor.ProcessorABC):
    def __init__(self, samp_info, do_roccor=True, do_fsr=True, evaluate_dnn=False,\
                 do_timer=False, save_unbin=True, do_lheweights=True, do_geofit=False, apply_jec=True,\
                 do_jer=True, do_jecunc=False, do_nnlops=True, do_btagsf=False, debug=False): 
        from config.parameters import parameters
        from config.variables import variables
        if not samp_info:
            print("Samples info missing!")
            return
        self.samp_info = samp_info
        self.year = self.samp_info.year

#        self.mass_window = mass_window
        self.mass_window = [70, 150]
        self.do_roccor = do_roccor
        self.save_unbin = save_unbin
        self.do_geofit = do_geofit
        self.apply_jec = apply_jec
        self.do_jer = do_jer
        self.do_jecunc = do_jecunc
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
        accumulators = processor.dict_accumulator({'binned':processor.dict_accumulator({}), 'unbinned':processor.dict_accumulator({})})
                        
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
                
        self.vars_unbin = [v.name for v in variables]
        
        ### Prepare accumulators for binned output ###
        
        for v in variables:
            if v.name=='dimuon_mass':
                axis = hist.Bin(v.name, v.caption, v.nbins, self.mass_window[0], self.mass_window[1])
            else:
                axis = hist.Bin(v.name, v.caption, v.nbins, v.xmin, v.xmax)
            accumulators['binned'][v.name] = hist.Hist("Counts", dataset_axis, region_axis, channel_axis, syst_axis, axis)
            
#        accumulators['binned']['cutflow'] = processor.defaultdict_accumulator(int)

        ### Prepare accumulators for unbinned output ###
    
        
        ### --------------------------------------- ###
        
        #for p in self.datasets_to_save_unbin:
        if self.save_unbin:
            for p in self.samp_info.samples:
                for v in self.vars_unbin:
                    if 'dnn_score' in v: continue
                    for c in self.channels:
                        for r in self.regions:
    #                        if 'z-peak' in r: continue # don't need unbinned data for Z-peak
                            accumulators['unbinned'][f'{v}_unbin_{p}_c_{c}_r_{r}'] = processor.column_accumulator(np.ndarray([]))
                            # have to encode everything into the name because having multiple axes isn't possible
        
        self._accumulator = processor.dict_accumulator(accumulators)
    
        ### --------------------------------------- ###
        
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
#         Have to do the last line because of a bug in lookup_tools
#         For 1-dimensional histograms, _axes is a tuple, and np.searchsorted doesn't understand it
#         https://github.com/CoffeaTeam/coffea/blob/2650ad7657094f6e50ebf962a1fc1763cd2c6601/coffea/lookup_tools/dense_lookup.py#L37
#         TODO: tell developers?

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
            JECcorrector_Data = FactorizedJetCorrector(**{name: Jetevaluator[name] for name in self.parameters['jec_names_data'][run]})
            JECuncertainties_Data = JetCorrectionUncertainty(**{name:Jetevaluator[name] for name in self.parameters['junc_names_data'][run]})

            self.Jet_transformer_data[run] = JetTransformer(jec=JECcorrector_Data,junc=JECuncertainties_Data)
            
        all_jec_names = [name for name in dir(Jetevaluator) if self.parameters['jec_unc_sources'] in name]
        self.JECuncertaintySources =\
        JetCorrectionUncertainty(**{name: Jetevaluator[name] for name in all_jec_names})
        self.jet_unc_names = list(self.JECuncertaintySources.levels)
#        print(self.jet_unc_names)
        if self.do_btagsf:
            self.btag_sf = BTagScaleFactor(self.parameters["btag_sf_csv"], BTagScaleFactor.RESHAPE, 'iterativefit,iterativefit,iterativefit')
    
    @property
    def accumulator(self):
        return self._accumulator
    
    @property
    def columns(self):
        return self._columns
    
    def process(self, df):
        # TODO: btag sf
        # TODO: Add systematic uncertainties
        
        # Variables to add (all need to be saved for unbinned analysis):
        # dimuon_phiCS

        hlt = np.zeros(df.shape[0], dtype=bool)
        bad_hlt = False
        for hlt_path in self.parameters['hlt']:
            if hlt_path in df.HLT.columns:
                hlt = hlt | df.HLT[hlt_path]

        
        df = df[hlt&(df.Muon.counts>1)]
        if self.debug:
            print("Events loaded: ", len(df))
            all_muons = len(df.Muon.flatten())
            
        if self.timer:
            self.timer.update()
            
        output = self.accumulator.identity()
        dataset = df.metadata['dataset']
        isData = 'data' in dataset
            
        nEvts = df.shape[0]

        weights = pd.DataFrame(1, index=np.arange(nEvts), columns=['nominal'])
        
        if isData:
            lumi_info = LumiMask(self.parameters['lumimask'])
            lumimask = lumi_info(df.run, df.luminosityBlock)
            event_weight = np.ones(nEvts)
            good_run = np.zeros(nEvts, dtype=bool)
        else:    
            lumimask = np.ones(nEvts, dtype=bool)
            genweight = df.genWeight.flatten()
            if '2016' in self.year:
                pu_weight = self.puLookup(dataset, df.Pileup.nTrueInt)
                pu_weight_up = self.puLookup_Up(dataset, df.Pileup.nTrueInt)
                pu_weight_down = self.puLookup_Down(dataset, df.Pileup.nTrueInt)
            else:
                pu_weight = self.puLookup(df.Pileup.nTrueInt)
                pu_weight_up = self.puLookup_Up(df.Pileup.nTrueInt)
                pu_weight_down = self.puLookup_Down(df.Pileup.nTrueInt)
                
            event_weight = genweight*pu_weight
            weights = weights.multiply(genweight, axis=0)
#            if self.debug:
#                print('Avg. pu weight: ', pu_weight.mean())
            add_systematic(weights, 'pu_weight', pu_weight, pu_weight_up, pu_weight_down)
            if dataset in self.lumi_weights:
                event_weight = event_weight*self.lumi_weights[dataset]
                weights = weights.multiply(self.lumi_weights[dataset], axis=0)
            if self.parameters["do_l1prefiring_wgts"]:
                event_weight = event_weight*df.L1PreFiringWeight.Nom.flatten()
                add_systematic(weights, 'l1prefiring_weight', df.L1PreFiringWeight.Nom.flatten(),\
                               df.L1PreFiringWeight.Up.flatten(), df.L1PreFiringWeight.Dn.flatten()) 

#        if self.evaluate_dnn:
#            ev_num_filter = ((df.event%2)==0)
#        else:
#            ev_num_filter = ((df.event%2)==1)
            
#        mask = hlt & lumimask & ev_num_filter

        mask = lumimask       
    
        # Filter 0: HLT & lumimask
        #--------------------------------#    
        df = df[mask]
        event_weight = event_weight[mask]
        weights = weights[mask]
        if self.timer:
            self.timer.add_checkpoint("Applied HLT and lumimask")
       #--------------------------------# 

        mu = df.Muon
        muons_pt = mu.pt.flatten()
        muons_eta = mu.eta.flatten()
        muons_phi = mu.phi.flatten()
        muons_mass = mu.mass.flatten()
        muons_iso = mu.pfRelIso04_all.flatten()          
            
        if self.do_fsr:
            fsr = df.FsrPhoton
            muons_fsrIndex = mu.fsrPhotonIdx.flatten()
            muons_offsets = mu.counts2offsets(mu.counts)
            fsr_pt = fsr.pt.flatten()
            fsr_eta = fsr.eta.flatten()
            fsr_phi = fsr.phi.flatten()
            fsr_offsets = fsr.counts2offsets(fsr.counts)
#            print(type(fsr_pt))
#            print(type(fsr_eta))
#            print(type(fsr_phi))
            muons_pt, muons_eta, muons_phi, muons_mass, muons_iso = correct_muon_with_fsr(muons_offsets, fsr_offsets,\
                                  muons_pt, muons_eta, muons_phi,\
                                  muons_mass, muons_iso,\
                                  muons_fsrIndex, fsr_pt, fsr_eta,\
                                  fsr_phi) 
        if self.do_roccor:
            muons_pt = muons_pt*apply_roccor(self.rochester, isData, mu).flatten()

        # GeoFit correction
        if self.do_geofit:
            muons_dxybs = mu.dxybs.flatten()
            muons_charge = mu.charge.flatten()
            muons_pt = apply_geofit(muons_pt, muons_eta, muons_dxybs, muons_charge, self.year).flatten() 
            
        updated_attrs = {'pt': muons_pt, 'eta':muons_eta, 'phi':muons_phi, 'mass':muons_mass, 'pfRelIso04_all':muons_iso}

        muonarrays = {key:mu[key].flatten() for key in mu.columns}
        muonarrays.update(updated_attrs)
        muons = JaggedCandidateArray.candidatesfromcounts(mu.counts, **muonarrays)
            
        pass_event_flags = np.ones(df.shape[0], dtype=bool)
        for flag in self.parameters["event_flags"]:
            pass_event_flags = pass_event_flags & df.Flag[flag]
        
        pass_muon_flags = np.ones(df.shape[0], dtype=bool)
        for flag in self.parameters["muon_flags"]:
            pass_muon_flags = pass_muon_flags & muons[flag]
            
        muons_for_jet_selection = muons[(abs(muons.eta) < self.parameters["muon_eta_cut"]) &\
                        (muons.pfRelIso04_all < self.parameters["muon_iso_cut"]) &\
                        muons[self.parameters["muon_id"]]
                     ]
            
        muons = muons[(muons.pt > self.parameters["muon_pt_cut"]) & (abs(muons.eta) < self.parameters["muon_eta_cut"]) &\
                        (muons.pfRelIso04_all < self.parameters["muon_iso_cut"]) &\
                        muons[self.parameters["muon_id"]] & pass_muon_flags]    
        


        two_os_muons = ((muons.counts == 2) & (muons['charge'].prod() == -1))
        
        electrons = df.Electron[(df.Electron.pt > self.parameters["electron_pt_cut"]) &\
                                     (abs(df.Electron.eta) < self.parameters["electron_eta_cut"]) &\
                                     (df.Electron[self.parameters["electron_id"]] == 1)]
                
        electron_veto = (electrons.counts==0)
        good_pv = (df.PV.npvsGood > 0)
        event_filter = (pass_event_flags & two_os_muons & electron_veto & good_pv).flatten()
#        if self.debug:
#            event_filter = np.ones(len(event_weight), dtype=bool)
        # Filter 1: Event selection
        #--------------------------------#    
        df = df[event_filter]
        muons = muons[event_filter]
        muons_for_jet_selection = muons_for_jet_selection[event_filter]
        event_weight = event_weight[event_filter]
        weights = weights[event_filter]
        if self.timer:
            self.timer.add_checkpoint("Applied preselection")
        #--------------------------------#  
        
        muons_jca = JaggedCandidateArray.candidatesfromcounts(
            muons.counts,
            pt=muons.pt.flatten(),
            eta=muons.eta.flatten(),
            phi=muons.phi.flatten(),
            mass=muons.mass.flatten(),
            charge=muons.charge.flatten(),
        )
        
        dimuons = muons_jca.distincts()
        
        mu1 = muons[muons.pt.argmax()]
        mu2 = muons[muons.pt.argmin()]

        if self.do_nnlops and ('ggh' in dataset):
            nnlops = NNLOPS_Evaluator('data/NNLOPS_reweight.root')
            nnlopsw = np.ones(len(event_weight))
            gen_njets = df.GenJet[df.GenJet.pt>30.].counts
            gen_higgs = df.GenPart[(df.GenPart.pdgId == 25)&(df.GenPart.status == 62)]
            has_higgs = gen_higgs.counts>0
            gen_hpt = np.zeros(len(event_weight))
            gen_hpt[has_higgs] = gen_higgs[has_higgs].pt[:,0]
            if 'amc' in dataset:
                nnlopsw[has_higgs] = nnlops.evaluate(gen_hpt[has_higgs], gen_njets[has_higgs], "mcatnlo")
            elif 'powheg' in dataset:
                nnlopsw[has_higgs] = nnlops.evaluate(gen_hpt[has_higgs], gen_njets[has_higgs], "powheg")
            event_weight = event_weight * nnlopsw
            weights = weights.multiply(nnlopsw, axis=0)
            
        if 'dy' in dataset:
            zpt_weights = self.evaluator[self.zpt_path](dimuons.pt).flatten()
            event_weight = event_weight*zpt_weights
            weights = weights.multiply(zpt_weights, axis=0)
#            if self.debug:
#                print('Avg. zpt weight: ', zpt_weights.mean())

        mu_pass_leading_pt = muons[(muons.pt > self.parameters["muon_leading_pt"]) &\
                                   (muons.pfRelIso04_all < self.parameters["muon_trigmatch_iso"]) &\
                                   muons[self.parameters["muon_trigmatch_id"]]]
        
#        if self.debug:
#            print("Muon sel. eff.: ", len(mu_pass_leading_pt.flatten())/all_muons)

        trigmuarrays = {key:df.TrigObj[key].flatten() for key in df.TrigObj.columns}
        trigmuarrays.update({'mass':0})
        trig_muons = JaggedCandidateArray.candidatesfromcounts(df.TrigObj.counts, **trigmuarrays)

        trig_muons = trig_muons[trig_muons.id == 13]
        muTrig = mu_pass_leading_pt.cross(trig_muons, nested = True)
        _,_,dr = delta_r(muTrig.i0.eta, muTrig.i1.eta, muTrig.i0.phi, muTrig.i1.phi)
        matched = (dr < self.parameters["muon_trigmatch_dr"])


        # at least one muon matched with L3 object, and that muon passes pt, iso and id cuts
        trig_matched = (mu_pass_leading_pt[matched.any()].counts>0)

        dimuon_filter = ((mu1.pt>self.parameters["muon_leading_pt"]) &\
                         trig_matched &\
                         (dimuons.mass > self.mass_window[0]) & (dimuons.mass < self.mass_window[1])).flatten()

        if not isData:
            muID = np.ones(len(muons.flatten()), dtype=float)
            muIso = np.ones(len(muons.flatten()), dtype=float)
            muTrig = np.ones(len(muons.flatten()), dtype=float)
            muIDerr = np.zeros(len(muons.flatten()), dtype=float)
            muIsoerr = np.zeros(len(muons.flatten()), dtype=float)
            muTrigerr = np.zeros(len(muons.flatten()), dtype=float)
            if '2016' in self.year:
                muID = self.mu_id_sf(muons.eta.compact(), muons.pt.compact())
                muIso = self.mu_iso_sf(muons.eta.compact(), muons.pt.compact())
                muTrig_data = self.mu_trig_eff_data(abs(muons.eta.compact()), muons.pt.compact())
                muTrig_mc = self.mu_trig_eff_mc(abs(muons.eta.compact()), muons.pt.compact())
                muIDerr = self.mu_id_err(muons.eta.compact(), muons.pt.compact())
                muIsoerr = self.mu_iso_err(muons.eta.compact(), muons.pt.compact())
                muTrigerr_data = self.mu_trig_err_data(abs(muons.eta.compact()), muons.pt.compact())
                muTrigerr_mc = self.mu_trig_err_mc(abs(muons.eta.compact()), muons.pt.compact())
            elif '2017' in self.year:
                muID = self.mu_id_sf(muons.pt.compact(), abs(muons.eta.compact()))
                muIso = self.mu_iso_sf(muons.pt.compact(), abs(muons.eta.compact()))
                muTrig_data = self.mu_trig_eff_data(abs(muons.eta.compact()), muons.pt.compact())
                muTrig_mc = self.mu_trig_eff_mc(abs(muons.eta.compact()), muons.pt.compact())                
                muIDerr = self.mu_id_err(muons.pt.compact(), abs(muons.eta.compact()))
                muIsoerr = self.mu_iso_err(muons.pt.compact(), abs(muons.eta.compact()))
                muTrigerr_data = self.mu_trig_err_data(abs(muons.eta.compact()), muons.pt.compact())
                muTrigerr_mc = self.mu_trig_err_mc(abs(muons.eta.compact()), muons.pt.compact())
            elif '2018' in self.year:
                muID = self.mu_id_sf(muons.pt.compact(), abs(muons.eta.compact()))
                muIso = self.mu_iso_sf(muons.pt.compact(), abs(muons.eta.compact()))
                muTrig_data = self.mu_trig_eff_data(abs(muons.eta.compact()), muons.pt.compact())
                muTrig_mc = self.mu_trig_eff_mc(abs(muons.eta.compact()), muons.pt.compact())
                muIDerr = self.mu_id_err(muons.pt.compact(), abs(muons.eta.compact()))
                muIsoerr = self.mu_iso_err(muons.pt.compact(), abs(muons.eta.compact()))
                muTrigerr_data = self.mu_trig_err_data(abs(muons.eta.compact()), muons.pt.compact())
                muTrigerr_mc = self.mu_trig_err_mc(abs(muons.eta.compact()), muons.pt.compact())
                
            muTrig = (1 - (1. - muTrig_data).prod()) / (1 - (1. - muTrig_mc).prod())
            muTrig_up = (1 - (1. - muTrig_data - muTrigerr_data).prod()) / (1 - (1. - muTrig_mc - muTrigerr_mc).prod())
            muTrig_down = (1 - (1. - muTrig_data + muTrigerr_data).prod()) / (1 - (1. - muTrig_mc + muTrigerr_mc).prod())

            muSF = (muID*muIso*muTrig).prod()
            muSF_up = ((muID + muIDerr) * (muIso + muIsoerr) * muTrig_up).prod()
            muSF_down = ((muID - muIDerr) * (muIso - muIsoerr) * muTrig_down).prod()
            event_weight = event_weight*muSF
            add_systematic(weights, 'muSF', muSF, muSF_up, muSF_down)
#            if self.debug:
#                print('Avg. muID: ', muID.mean())
#                print('Avg. muIso: ', muIso.mean())
#                print('Avg. muTrig: ', muTrig.mean()) 
#                print('Avg. muTrig up: ', muTrig_up.mean()) 
#                print('Avg. muTrig down: ', muTrig_down.mean()) 
#                print('Avg. muSF: ', muSF.mean())

#        if self.debug:
#            dimuon_filter = np.ones(len(event_weight), dtype=bool)
    
        # Filter 2: Dimuon pair selection
        #--------------------------------#
        df = df[dimuon_filter]   
        mu1 = mu1[dimuon_filter] 
        mu2 = mu2[dimuon_filter] 
        muons = muons[dimuon_filter]
        muons_for_jet_selection = muons_for_jet_selection[dimuon_filter]
        dimuons = dimuons[dimuon_filter]
        event_weight = event_weight[dimuon_filter]
        weights = weights[dimuon_filter]
        if self.timer:
            self.timer.add_checkpoint("Applied dimuon cuts")
        #--------------------------------#

        jetarrays = {key:df.Jet[key].flatten() for key in df.Jet.columns} 
        raw_arrays = {'ptRaw':(df.Jet.pt * (1-df.Jet.rawFactor)).flatten(),\
                          'massRaw':(df.Jet.mass * (1-df.Jet.rawFactor)).flatten()}
        jetarrays.update(raw_arrays)
        
        if not isData:
            ptGenJet = df.Jet.pt.zeros_like()
            genJetIdx = df.Jet.genJetIdx
            mask = genJetIdx<df.GenJet.counts
            ptGenJet[mask] = df.GenJet[genJetIdx[mask]].pt
            gen_arrays = {'ptGenJet':ptGenJet.flatten()}
            jetarrays.update(gen_arrays)
            
        jet = JaggedCandidateArray.candidatesfromcounts(df.Jet.counts, **jetarrays)
        jet['rho'] = jet.pt.ones_like()*df.fixedGridRhoFastjetAll

        
        # Jet ID
        if "loose" in self.parameters["jet_id"]:
            jet_id = (jet.jetId >= 1)
        elif "tight" in self.parameters["jet_id"]:
            if '2016' in self.year:
                jet_id = (jet.jetId >= 3)
            else:
                jet_id = (jet.jetId >= 2)
        else:
            jet_id = jet.ones_like()

        # Jet PU ID
        jet_puid_opt = self.parameters["jet_puid"]
        jet_puid_wps = {
            "loose": (((jet.puId >= 4) & (jet.pt < 50)) | (jet.pt > 50)),
            "medium": (((jet.puId >= 6) & (jet.pt < 50)) | (jet.pt > 50)),
            "tight": (((jet.puId >= 7) & (jet.pt < 50)) | (jet.pt > 50)),
        }
        if jet_puid_opt in ["loose", "medium", "tight"]:
            jet_puid = jet_puid_wps[jet_puid_opt]
        elif "2017corrected" in jet_puid_opt:
            eta_window = ((abs(jet.eta)>2.6)&(abs(jet.eta)<3.0))
            not_eta_window = ((abs(jet.eta)<2.6)|(abs(jet.eta)>3.0))
            jet_puid = (eta_window & (jet.puId >= 7)) | (not_eta_window & jet_puid_wps['loose'])
            jet_puid_opt = "loose" # for sf evaluation
        else:
            jet_puid = jet.ones_like()

        # Jet PUID scale factors
        wp_dict = {"loose": "L", "medium": "M", "tight": "T"}
        wp = wp_dict[jet_puid_opt]
        h_eff_name = f"h2_eff_mc{self.year}_{wp}"
        h_sf_name = f"h2_eff_sf{self.year}_{wp}"
        puid_eff = self.evaluator[h_eff_name](jet.pt, jet.eta)
        puid_sf = self.evaluator[h_sf_name](jet.pt, jet.eta)
        jets_passed = (jet.pt>25) & (jet.pt<50) & jet_puid
        jets_failed = (jet.pt>25) & (jet.pt<50) & (~jet_puid)
        
        pMC   = puid_eff[jets_passed].prod() * (1.-puid_eff[jets_failed]).prod() 
        pData = puid_eff[jets_passed].prod() * puid_sf[jets_passed].prod() * \
                (1. - puid_eff[jets_failed] * puid_sf[jets_failed]).prod()
            
        puid_weight = np.ones(len(event_weight))
        puid_weight[pMC!=0] = np.divide(pData[pMC!=0], pMC[pMC!=0])
        if not isData:
            event_weight = event_weight * puid_weight
            weights = weights.multiply(puid_weight, axis=0)

        # Btag weight    
        if self.do_btagsf and not isData:
            btag_wgt = self.btag_sf('central', jet.hadronFlavour, abs(jet.eta), jet.pt, jet.btagDeepB, True)
            print(btag_wgt.prod().mean())
        
#        if self.debug:
#            print("Avg. jet PU ID SF: ", puid_weight.mean())
        
        
        mujet = jet.cross(muons_for_jet_selection, nested=True)
        _,_,deltar_mujet = delta_r(mujet.i0.eta, mujet.i1.eta, mujet.i0.phi, mujet.i1.phi)

        deltar_mujet_ok =  (deltar_mujet > self.parameters["min_dr_mu_jet"]).all()
        
        good_jet_id = jet_id & jet_puid & (jet.qgl > -2) & deltar_mujet_ok

        jet = jet[good_jet_id]
        
#        if self.debug:
#            print("Jet sel. eff. (ID): ", len(jet.flatten())/len(good_jet_id.flatten()))
        original_jet_pt = jet.pt.flatten()
    
    
        if self.apply_jec: 

## Caltech implementation: may be useful if custom JER recipe is proposed            
#            from python.JetTransformer_caltech import JetTransformerCaltech, jetmet_corrections
#            run_name = ''
#            if isData:
#                for run in self.data_runs: # 'B', 'C', 'D', etc...
#                    if run in dataset: # dataset name is something like 'data_B'
#                        run_name = run
        
#            jetmet_corr = jetmet_corrections[self.year][self.parameters["jec_tag"]]
        
#            jet_tr_caltech = JetTransformerCaltech(jet, run_name, jetmet_corr, isData, self.do_jer)
#            pt_corr = jet_tr_caltech.pt_jec
            
#            if self.do_jer:
#                pt_corr=jet_tr_caltech.pt_jec_jer
#            else:
#                pt_corr=jet_tr_caltech.pt_jec

#            if self.debug:
#                print("mean jec pt change: ", (pt_corr-original_jet_pt).mean())
#                print("std jec pt change: ", (pt_corr-original_jet_pt).std())
                
            cols = {
                '__fast_pt': 'pt',
                '__fast_eta': 'eta',
                '__fast_phi': 'phi',
                '__fast_mass': 'mass',
                'qgl': 'qgl',
                'btagDeepB':'btagDeepB',
                'ptRaw':'ptRaw',
                'massRaw':'massRaw',
                'rho':'rho',
                'area':'area',
            }
        
            if not isData:
                cols.update({'genJetIdx': 'genJetIdx', 'ptGenJet':'ptGenJet'})


            jetarrays = {v:jet[key].flatten() for key, v in cols.items()}
#            jetarrays.update({'pt': pt_corr})            
            jet = JaggedCandidateArray.candidatesfromcounts(jet.counts, **jetarrays)
            jet_variation_names = ['nominal']
            
            if self.do_jecunc:
                for junc_name in self.jet_unc_names:
                    if junc_name not in self.parameters["jec_unc_to_consider"]: continue
                    jet_variation_names += [f"{junc_name}_up", f"{junc_name}_down"]
            
            if isData:
                for run in self.data_runs: # 'B', 'C', 'D', etc...
                    if run in dataset: # dataset name is something like 'data_B'
                        self.Jet_transformer_data[run].transform(jet, forceStochastic=False)            
            else:
                self.Jet_transformer.transform(jet, forceStochastic=False)
                if self.do_jecunc:
                    for junc_name in self.jet_unc_names:
                        if junc_name not in self.parameters["jec_unc_to_consider"]: continue
                        jec_up_down = get_jec_unc(junc_name, jet.pt, jet.eta, self.JECuncertaintySources)
                        jec_corr_up, jec_corr_down = jec_up_down[:,:,0], jec_up_down[:,:,1]
                        pt_name_up = f"pt_{junc_name}_up"
                        pt_name_down = f"pt_{junc_name}_down"
                        jet.add_attributes(**{pt_name_up: jet.pt*jec_corr_up, pt_name_down: jet.pt*jec_corr_down})
                        
    
#            if self.debug:
#                print("mean jec pt change (Coffea): ", (jet.pt.flatten()-original_jet_pt).mean())
#                print("std jec pt change (Coffea): ", (jet.pt.flatten()-original_jet_pt).std())                   

        # Jet selection
        jet_selection = ((jet.pt > self.parameters["jet_pt_cut"]) &\
                         (abs(jet.eta) < self.parameters["jet_eta_cut"]))

        jets = jet[jet_selection]
        has_good_jets = np.zeros(len(event_weight), dtype=bool)
        has_good_jets[jets.counts>0] = (jets[jets.counts>0][:,0]["__fast_pt"]>35.)
        
        # Separate from ttH and VH phase space        
        nBtagLoose = jets[(jets.btagDeepB>self.parameters["btag_loose_wp"]) & (abs(jets.eta)<2.5)].counts
        nBtagMedium = jets[(jets.btagDeepB>self.parameters["btag_medium_wp"])  & (abs(jets.eta)<2.5)].counts
        jet_filter = has_good_jets&((nBtagLoose<2)&(nBtagMedium<1))
        # Filter 3: Jet filter
        #--------------------------------#
        df = df[jet_filter]   
        mu1 = mu1[jet_filter] 
        mu2 = mu2[jet_filter] 
        muons = muons[jet_filter]
        dimuons = dimuons[jet_filter]
        jets = jets[jet_filter]
        jet_selection = jet_selection[jet_filter]
        event_weight = event_weight[jet_filter]
        weights = weights[jet_filter]
        if self.timer:
            self.timer.add_checkpoint("Applied jet cuts")

        #--------------------------------#

        
        dpt1 = (mu1.ptErr*dimuons.mass) / (2*mu1.pt)
        dpt2 = (mu2.ptErr*dimuons.mass) / (2*mu2.pt)
        if isData:
            label = f"res_calib_Data_{self.year}"
        else:
            label = f"res_calib_MC_{self.year}"
        calibration = np.array(self.evaluator[label](mu1.pt.flatten(), abs(mu1.eta.flatten()), abs(mu2.eta.flatten())))
        dimuon_mass_res = np.sqrt(dpt1 * dpt1 + dpt2 * dpt2) * calibration
        dimuon_mass_res_rel = dimuon_mass_res / dimuons.mass

        
        # In the computations below I'm trying to keep the size of objects in first dimension the same
        # as it is in the previous step, in order to be able to apply event_weight similarly for all variables

        one_jet = ((jet_selection).any() & (jets.counts>0))
        two_jets = ((jet_selection).any() & (jets.counts>1))
        
        category = np.empty(df.shape[0], dtype=object)
        category[~two_jets] = 'ggh_01j'
        
        jet1_mask = np.zeros(len(event_weight))
        jet1_mask[one_jet] = 1
        jet2_mask = np.zeros(len(event_weight))
        jet2_mask[two_jets] = 1
        
        cols = {
            '__fast_pt': 'pt',
            '__fast_eta': 'eta',
            '__fast_phi': 'phi',
            '__fast_mass': 'mass',
            'qgl': 'qgl',
        }
        
        if not isData:
            cols.update({'genJetIdx': 'genJetIdx'})
            if self.do_jecunc:
                for v_name in jet_variation_names:
                    if v_name=='nominal': continue
                    cols.update({f"pt_{v_name}": f"pt_{v_name}"})
      
        my_jet1_arrays = {v:jets[one_jet][jets[one_jet].pt.argmax()][key].flatten() for key, v in cols.items()}
        my_jet2_arrays = {v:jets[two_jets][jets[two_jets].pt.argmin()][key].flatten() for key, v in cols.items()}
                
        jet1 = JaggedCandidateArray.candidatesfromcounts(jet1_mask, **my_jet1_arrays)
        jet2 = JaggedCandidateArray.candidatesfromcounts(jet2_mask, **my_jet2_arrays)

        if not isData:     
            genJetMass = np.zeros(len(event_weight), dtype=float)
            genJetMass_filtered = genJetMass[two_jets]
            idx1 = jet1.genJetIdx[two_jets].flatten()
            idx2 = jet2.genJetIdx[two_jets].flatten()
            gjets = df.GenJet[two_jets]
            has_gen_pair = (idx1 >= 0) & (idx2 >= 0) & (idx1 < gjets.counts) & (idx2 < gjets.counts)
            _,_,_, genJetMass_filtered[has_gen_pair] = p4_sum(gjets[has_gen_pair,idx1[has_gen_pair]],\
                                                               gjets[has_gen_pair,idx2[has_gen_pair]])
            genJetMass[two_jets] = genJetMass_filtered

        dijet_pairs = jets[two_jets, 0:2]
        dijet_mask = np.zeros(len(event_weight))
        dijet_mask[two_jets] = 2
        dijet_jca = JaggedCandidateArray.candidatesfromcounts(
            dijet_mask,
            pt=dijet_pairs.pt.flatten(),
            eta=dijet_pairs.eta.flatten(),
            phi=dijet_pairs.phi.flatten(),
            mass=dijet_pairs.mass.flatten(),
        )
        
        dijet = dijet_jca.distincts()
        dijet = dijet.p4.sum()

        dijet_deta = np.full(len(event_weight), -999.)
        dijet_deta[two_jets] = abs(jet1[two_jets].eta - jet2[two_jets].eta)
        
        dijet_dphi = np.full(len(event_weight), -999.)
        dijet_dphi[two_jets] = abs(jet1[two_jets].p4.delta_phi(jet2[two_jets].p4))

        zeppenfeld = np.full(len(event_weight), -999.)
        zeppenfeld[two_jets] = (dimuons.eta[two_jets] - 0.5*(jet1.eta[two_jets] + jet2.eta[two_jets]))
        
        rpt = np.full(len(event_weight), -999.)
        mmjj_pt = np.full(len(event_weight), 0.)
        mmjj_eta = np.full(len(event_weight), -999.)
        mmjj_phi = np.full(len(event_weight), -999.)
        mmjj_mass = np.full(len(event_weight), 0.)
        mmjj_pt[two_jets], mmjj_eta[two_jets], mmjj_phi[two_jets], mmjj_mass[two_jets] = p4_sum(dimuons[two_jets], dijet[two_jets])
        rpt[two_jets] =  mmjj_pt[two_jets]/(dimuons.pt[two_jets] + jet1.pt[two_jets] + jet2.pt[two_jets])
        

        # This might be not needed
        
        sjarrays = {key:df.SoftActivityJet[key].flatten() for key in df.SoftActivityJet.columns}   
        sjarrays.update({'mass':0})
        softjets = JaggedCandidateArray.candidatesfromcounts(df.SoftActivityJet.counts, **sjarrays)

        nsoftjets2 = df.SoftActivityJetNjets2
        nsoftjets5 = df.SoftActivityJetNjets5
        htsoft2 = df.SoftActivityJetHT2
        htsoft5 = df.SoftActivityJetHT5
        
        # Events with 0 selected jets: clear soft jets from muons
        no_jets = ~one_jet
        
        sj_mm0 = softjets[no_jets].cross(muons[no_jets], nested=True)
        _,_,dr_mm0 = delta_r(sj_mm0.i0.eta, sj_mm0.i1.eta, sj_mm0.i0.phi, sj_mm0.i1.phi)
        bad_sj_mm0 = (dr_mm0*dr_mm0 < self.parameters["softjet_dr2"])
        bad_sj0 = (bad_sj_mm0.any()).all()
 
        sj2_sel0 = np.zeros(len(event_weight), dtype=bool)
        sj2_sel0[no_jets] = ((softjets.pt[no_jets]>2.) & bad_sj0).any()
        nsoftjets2[sj2_sel0] = nsoftjets2[sj2_sel0] - softjets[sj2_sel0].counts
        htsoft2[sj2_sel0] = htsoft2[sj2_sel0] - softjets.pt[sj2_sel0].sum()
 
        sj5_sel0 = np.zeros(len(event_weight), dtype=bool)
        sj5_sel0[no_jets] = ((softjets.pt[no_jets]>5.) & bad_sj0).any()
        nsoftjets5[sj5_sel0] = nsoftjets5[sj5_sel0] - softjets[sj5_sel0].counts
        htsoft5[sj5_sel0] = htsoft5[sj5_sel0] - softjets.pt[sj5_sel0].sum()
        
        # Events with exactly 1 selected jet: clear soft jets from muons and the jet
        only_one_jet = one_jet & (~two_jets)
        
        sj_mm1 = softjets[only_one_jet].cross(muons[only_one_jet], nested=True)
        _,_,dr_mm1 = delta_r(sj_mm1.i0.eta, sj_mm1.i1.eta, sj_mm1.i0.phi, sj_mm1.i1.phi)
        bad_sj_mm1 = (dr_mm1*dr_mm1 < self.parameters["softjet_dr2"])
        
        sj_j1_1 = softjets[only_one_jet].cross(jet1[only_one_jet], nested=True)
        _,_,dr_j1_1 = delta_r(sj_j1_1.i0.eta, sj_j1_1.i1.eta, sj_j1_1.i0.phi, sj_j1_1.i1.phi)
        bad_sj_j1_1 = (dr_j1_1*dr_j1_1 < self.parameters["softjet_dr2"])
        
        bad_sj1 = (bad_sj_mm1.any() | bad_sj_j1_1).all()

        sj2_sel1 = np.zeros(len(event_weight), dtype=bool)
        sj2_sel1[only_one_jet] = ((softjets.pt[only_one_jet]>2.) & bad_sj1).any()
        nsoftjets2[sj2_sel1] = nsoftjets2[sj2_sel1] - softjets[sj2_sel1].counts
        htsoft2[sj2_sel1] = htsoft2[sj2_sel1] - softjets.pt[sj2_sel1].sum()
        
        sj5_sel1 = np.zeros(len(event_weight), dtype=bool)
        sj5_sel1[only_one_jet] = ((softjets.pt[only_one_jet]>5.) & bad_sj1).any()
        nsoftjets5[sj5_sel1] = nsoftjets5[sj5_sel1] - softjets[sj5_sel1].counts
        htsoft5[sj5_sel1] = htsoft5[sj5_sel1] - softjets.pt[sj5_sel1].sum()
        
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

        bad_sj2 = (bad_sj_mm2.any() | bad_sj_j1_2 | bad_sj_j2).all()

        eta_sj_j1 = (sj_j1_2.i0.eta > sj_j1_2.i1.eta).all()
        eta_sj_j2 = (sj_j2.i0.eta > sj_j2.i1.eta).all()
        eta_j1j2 = (sj_j1_2.i1.eta > sj_j2.i1.eta).all()

        eta_sel = ( (eta_sj_j1 | ~eta_sj_j2) & eta_j1j2 ) | ( (~eta_sj_j1 | eta_sj_j2) & ~eta_j1j2 )

        sj2_sel2 = np.zeros(len(event_weight), dtype=bool)
        sj2_sel2[two_jets] = (((softjets.pt[two_jets]>2.) | eta_sel) & bad_sj2).any()
        nsoftjets2[sj2_sel2] = nsoftjets2[sj2_sel2] - softjets[sj2_sel2].counts
        htsoft2[sj2_sel2] = htsoft2[sj2_sel2] - softjets.pt[sj2_sel2].sum()
        
        sj5_sel2 = np.zeros(len(event_weight), dtype=bool)
        sj5_sel2[two_jets] = ((softjets.pt[two_jets]>5.) & bad_sj2).any()
        nsoftjets5[sj5_sel2] = nsoftjets5[sj5_sel2] - softjets[sj5_sel2].counts
        htsoft5[sj5_sel2] = htsoft5[sj5_sel2] - softjets.pt[sj5_sel2].sum()

        
        ###
        
        vbf_cut = (dijet.mass>400)&(dijet_deta>2.5)
        category[two_jets&(~vbf_cut)] = 'ggh_2j'
        category[two_jets&vbf_cut] = 'vbf'
        if self.debug:
#            print(f"{dataset}: ", event_weight[two_jets].sum(), "events w/ 2 jets")
            print(f"{dataset}: ", event_weight[two_jets&vbf_cut].sum(), "events in VBF")
        if self.timer:
            self.timer.add_checkpoint("Computed jet variables")
        
        assert(np.count_nonzero(category)==category.shape[0]), "Not all events have been categorized!"
        channels = list(set(category))
        
#----------------------------------------------------------------------------#
        # flatten variables where exactly one value per event expected

        deta_mumu, dphi_mumu, dr_mumu = delta_r(mu1.eta.flatten(), mu2.eta.flatten(), mu1.phi.flatten(), mu2.phi.flatten())
        deta_mumuj1 = np.zeros(len(event_weight))
        dphi_mumuj1 = np.zeros(len(event_weight))
        dr_mumuj1 = np.zeros(len(event_weight))
        
        deta_mumuj1[one_jet], dphi_mumuj1[one_jet], dr_mumuj1[one_jet] = delta_r(dimuons[one_jet].eta.flatten(),\
                                                                              jet1.eta.flatten(),\
                                                                              dimuons[one_jet].phi.flatten(),\
                                                                              jet1.phi.flatten())
        deta_mumuj2 = np.zeros(len(event_weight))
        dphi_mumuj2 = np.zeros(len(event_weight))
        dr_mumuj2 = np.zeros(len(event_weight))
        
        deta_mumuj2[two_jets], dphi_mumuj2[two_jets], dr_mumuj2[two_jets] = delta_r(dimuons[two_jets].eta.flatten(),\
                                                                              jet2.eta.flatten(),\
                                                                              dimuons[two_jets].phi.flatten(),\
                                                                              jet2.phi.flatten())

        min_deta_mumuj = np.zeros(len(event_weight))
        min_deta_mumuj[two_jets] = np.where(deta_mumuj1[two_jets], deta_mumuj2[two_jets],\
                                            (deta_mumuj1[two_jets] < deta_mumuj1[two_jets]))
        
        min_dphi_mumuj = np.zeros(len(event_weight))
        min_dphi_mumuj[two_jets] = np.where(dphi_mumuj1[two_jets], dphi_mumuj2[two_jets],\
                                            (dphi_mumuj1[two_jets] < dphi_mumuj1[two_jets]))
        
        mu1_px = mu1.pt*np.cos(mu1.phi)
        mu1_py = mu1.pt*np.sin(mu1.phi)
        mu1_pz = mu1.pt*np.sinh(mu1.eta)
        mu1_e  = np.sqrt(mu1_px**2 + mu1_py**2 + mu1_pz**2 + mu1.mass**2)
        mu2_px = mu2.pt*np.cos(mu2.phi)
        mu2_py = mu2.pt*np.sin(mu2.phi)
        mu2_pz = mu2.pt*np.sinh(mu2.eta)
        mu2_e  = np.sqrt(mu2_px**2 + mu2_py**2 + mu2_pz**2 + mu2.mass**2)
        
        cthetaCS = 2*(mu1_pz * mu2_e - mu1_e * mu2_pz) / (dimuons.mass * np.sqrt(dimuons.mass*dimuons.mass + dimuons.pt*dimuons.pt))
        
        variable_map = {
            'dimuon_mass': dimuons.mass.flatten(),
            'dimuon_mass_res': dimuon_mass_res.flatten(),
            'dimuon_mass_res_rel': dimuon_mass_res_rel.flatten(),
            'dimuon_pt': dimuons.pt.flatten(),
            'dimuon_eta': dimuons.eta.flatten(),
            'dimuon_phi': dimuons.phi.flatten(),
            'dimuon_dEta': deta_mumu,
            'dimuon_dPhi': dphi_mumu,
            'dimuon_dR': dr_mumu,
            'dimuon_cosThetaCS': cthetaCS.flatten(),
            
            'mu1_pt': mu1.pt.flatten(),
            'mu1_pt_over_mass': np.divide(mu1.pt.flatten(), dimuons.mass.flatten()),
            'mu1_eta': mu1.eta.flatten(),
            'mu1_phi': mu1.phi.flatten(),
            'mu1_iso': mu1.pfRelIso04_all.flatten(),

            'mu2_pt': mu2.pt.flatten().flatten(),
            'mu2_pt_over_mass': np.divide(mu2.pt.flatten(), dimuons.mass.flatten()),
            'mu2_eta': mu2.eta.flatten(),
            'mu2_phi': mu2.phi.flatten(),
            'mu2_iso': mu2.pfRelIso04_all.flatten(),
            
            'jet1_pt': jet1.pt,
            'jet1_eta': jet1.eta,
            'jet1_phi': jet1.phi,
            'jet1_qgl': jet1.qgl,
            'deta_mumuj1': deta_mumuj1,
            'dphi_mumuj1': dphi_mumuj1,
            
            'jet2_pt': jet2.pt,
            'jet2_eta': jet2.eta,
            'jet2_phi': jet2.phi,
            'jet2_qgl': jet2.qgl,
            'deta_mumuj2': deta_mumuj2,
            'dphi_mumuj2': dphi_mumuj2,
            
            'min_deta_mumuj': min_deta_mumuj,
            'min_dphi_mumuj': min_dphi_mumuj,
     
            'jj_mass': dijet.mass,
            'jj_pt': dijet.pt,
            'jj_eta': dijet.eta,
            'jj_phi': dijet.phi,
            'jj_dEta': dijet_deta,
            'jj_dPhi': dijet_dphi,      
 
            'mmjj_mass': mmjj_mass,
            'mmjj_pt': mmjj_pt,
            'mmjj_eta': mmjj_eta,
            'mmjj_phi': mmjj_phi,            
            'zeppenfeld': zeppenfeld,
            'rpt': rpt,
            
            'njets': jets.counts.flatten(),
            'nsoftjets2': nsoftjets2.flatten(),
            'nsoftjets5': nsoftjets5.flatten(),
            'htsoft2': htsoft2.flatten(),
            'htsoft5': htsoft5.flatten(),
        
            'npv': df.PV.npvsGood.flatten(),
            'met': df.MET.pt.flatten(),
            'event': df.event.flatten(),
            'event_weight': event_weight,
        }
        
        if ("dy" in dataset or "ewk" in dataset) and self.do_lheweights:
            for i in range(9):
                try:
                    variable_map[f'LHEScaleWeight_{i}'] = df.LHEScaleWeight[:,i]
                except:
                    variable_map[f'LHEScaleWeight_{i}'] = np.ones(len(event_weight), dtype=float)
        
#        print(len(event_weight))
        for syst in weights.columns:
            variable_map[f'weight_{syst}'] = np.array(weights[syst])
#            print(syst, len(np.array(weights[syst])))

        # Evaluate DNN 

        if self.evaluate_dnn:            
            from config.parameters import training_features
            #import keras.backend as K
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
                regions = get_regions(variable_map['dimuon_mass'])
            
                for v_name in jet_variation_names:
                    if (v_name != 'nominal') and (not self.do_jecunc): continue
                    if (v_name != 'nominal') and isData:
                        variable_map[f'dnn_score_{v_name}'] = variable_map['dnn_score_nominal']
                        continue
                    jet_vars = get_variated_jet_variables(v_name, jets, dimuons, one_jet, two_jets, event_weight)
                    dnn_score = np.full(len(event_weight), -1.)
                
                    for region, rcut in regions.items():
                        df_for_dnn = pd.DataFrame(columns=training_features)
                        mask = two_jets & rcut
                        n_rows = len(dimuons.mass[mask].flatten())
                        for trf in training_features:
                            if trf=='dimuon_mass' and region!='h-peak':
                                feature_column = np.full(sum(mask), 125.)
                            elif trf in jet_vars.keys():
                                feature_column = jet_vars[trf][mask]
                            else:
                                feature_column = variable_map[trf][mask]
                            assert(n_rows==len(feature_column))
                            df_for_dnn[trf] = feature_column
                        df_for_dnn[training_features] = (df_for_dnn[training_features]-scaler[0])/scaler[1]
                        try:
                            prediction = dnn_model.predict(df_for_dnn[training_features], verbose=0)
                            pred_array = tf.reshape(prediction, [-1]).eval()
                        except:
                            pred_array = np.zeros(sum(mask))

                        dnn_score[mask] = pred_array
                    
                    variable_map[f'dnn_score_{v_name}'] = np.arctanh((dnn_score))

            if self.timer:
                self.timer.add_checkpoint("Evaluated DNN")


        #################### Fill outputs ####################
        #------------------ Binned outputs ------------------#  
        for vname, expression in variable_map.items():
            if vname not in output['binned']: continue
            regions = get_regions(variable_map['dimuon_mass'])            
            for cname in self.channels:
                ccut = (category==cname)
                for rname, rcut in regions.items():
                    if (dataset in self.overlapping_samples) and (dataset not in self.specific_samples[rname][cname]): 
                        continue
#                    if isData and ('h-peak' in rname) and ('dimuon_mass' in vname): continue # blinding
                    if ('dy_m105_160_vbf_amc' in dataset) and ('vbf' in cname):
                        ccut = ccut & (genJetMass > 350.)
                    if ('dy_m105_160_amc' in dataset) and ('vbf' in cname):
                        ccut = ccut & (genJetMass < 350.)
                    value = expression[rcut & ccut]
                    if not value.size: continue # skip empty arrays
                    for syst in weights.columns:
                        if isinstance(value, awkward.JaggedArray):
                            #weight = event_weight[rcut & ccut][value.any()]
                            weight = weights[syst][rcut & ccut][value.any()]
                        else:
                            #weight = event_weight[rcut & ccut]
                            weight = weights[syst][rcut & ccut]
                        output['binned'][vname].fill(**{'dataset': dataset, 'region': rname, 'channel': cname, 'syst': syst,\
                                             vname: value.flatten(), 'weight': weight})

#        output['binned']['cutflow']['all events'] += nEvts
    

        
        #----------------- Unbinned outputs -----------------#
        
#        if dataset in self.datasets_to_save_unbin: # don't need unbinned data for all samples
        if self.save_unbin:
            for v in self.vars_unbin:
                if v not in variable_map: continue
                if 'dnn_score' in vname: continue 
                regions = get_regions(variable_map['dimuon_mass']) 
                for cname in self.channels:
                    ccut = (category==cname)
                    for rname, rcut in regions.items():
                        if ('dy_m105_160_vbf_amc' in dataset) and ('vbf' in cname):
                            ccut = ccut & (genJetMass > 350.)
                        if ('dy_m105_160_amc' in dataset) and ('vbf' in cname):
                            ccut = ccut & (genJetMass < 350.)
                        output['unbinned'][f'{v}_unbin_{dataset}_c_{cname}_r_{rname}'] += processor.column_accumulator(variable_map[v][rcut & ccut].flatten())
    
        if self.timer:
            self.timer.add_checkpoint("Filled outputs")
            
        if self.timer:
            self.timer.summary()

        return output
    
    def postprocess(self, accumulator):
        return accumulator
