from coffea import hist, util
from coffea.analysis_objects import JaggedCandidateArray, JaggedCandidateMethods
import coffea.processor as processor
from coffea.lookup_tools import extractor, dense_lookup, txt_converters, rochester_lookup
from coffea.lumi_tools import LumiMask

import awkward
import uproot
import numpy as np
import numba
import pandas as pd

from python.utils import apply_roccor, p4_sum, get_regions
from python.timer import Timer
from python.samples_info import SamplesInfo

def delta_r(eta1, eta2, phi1, phi2):
    deta = abs(eta1 - eta2)
    dphi = np.mod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi
    dr = np.sqrt(deta**2 + dphi**2)
    return deta, dphi, dr
    
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
                fsr_kin = {"pt": fsr_pt[ifsr], "eta": fsr_eta[ifsr], "phi": fsr_phi[ifsr],"mass": 0}

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
    def __init__(self, mass_window=[70,150], samp_info=SamplesInfo(), do_roccor=True, do_fsr=False, evaluate_dnn=True,\
                 do_timer=False): 
        from config.parameters import parameters
        from config.variables import variables, vars_unbin
        if not samp_info:
            print("Samples info missing!")
            return
        self.year = samp_info.year

        self.mass_window = mass_window
        self.do_roccor = do_roccor
        self.do_fsr = do_fsr
        self.evaluate_dnn = evaluate_dnn

        self.parameters = {k:v[self.year] for k,v in parameters.items()}
        self.timer = Timer('global') if do_timer else None
        
        self._columns = self.parameters["proc_columns"]

        dataset_axis = hist.Cat("dataset", "")
        region_axis = hist.Cat("region", " ") # Z-peak, Higgs SB, Higgs peak
        channel_axis = hist.Cat("channel", " ") # ggh or VBF       
        accumulators = {}
                        
        self.vars_unbin = vars_unbin
        self.regions = samp_info.regions
        self.channels = samp_info.channels

        self.overlapping_samples = samp_info.overlapping_samples
        self.specific_samples = samp_info.specific_samples
        self.datasets_to_save_unbin = samp_info.datasets_to_save_unbin
        self.lumi_weights = samp_info.lumi_weights
            
        ### Prepare accumulators for binned output ###
        
        for v in variables:
            if 'dimuon_mass' in v.name:
                axis = hist.Bin(v.name, v.caption, v.nbins, self.mass_window[0], self.mass_window[1])
            else:
                axis = hist.Bin(v.name, v.caption, v.nbins, v.xmin, v.xmax)
            accumulators[v.name] = hist.Hist("Counts", dataset_axis, region_axis, channel_axis, axis)
            
        accumulators['cutflow'] = processor.defaultdict_accumulator(int)

        ### Prepare accumulators for unbinned output ###
    
        
        ### --------------------------------------- ###
        
        for p in self.datasets_to_save_unbin:
            for v in self.vars_unbin:
                for c in self.channels:
                    for r in self.regions:
                        if 'z-peak' in r: continue # don't need unbinned data for Z-peak
                        accumulators[f'{v}_unbin_{p}_c_{c}_r_{r}'] = processor.column_accumulator(np.ndarray([]))
                        # have to encode everything into the name because having multiple axes isn't possible
        
        self._accumulator = processor.dict_accumulator(accumulators)
    
        ### --------------------------------------- ###
        
        mu_id_vals = 0
        mu_id_err = 0
        mu_iso_vals = 0
        mu_iso_err = 0
        mu_trig_vals = 0
        mu_trig_err = 0

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

            mu_trig_vals += trig_file[scaleFactors['trig'][1]].values * scaleFactors['scale']
            mu_trig_err += trig_file[scaleFactors['trig'][1]].variances**0.5 * scaleFactors['scale']
            mu_trig_edges = trig_file[scaleFactors['trig'][1]].edges

        self.mu_id_sf = dense_lookup.dense_lookup(mu_id_vals, mu_id_edges)
        self.mu_id_err = dense_lookup.dense_lookup(mu_id_err, mu_id_edges)
        self.mu_iso_sf = dense_lookup.dense_lookup(mu_iso_vals, mu_iso_edges)
        self.mu_iso_err = dense_lookup.dense_lookup(mu_iso_err, mu_iso_edges)
        self.mu_trig_sf = dense_lookup.dense_lookup(mu_trig_vals, mu_trig_edges)
        self.mu_trig_err = dense_lookup.dense_lookup(mu_trig_err, mu_trig_edges)    

        
        zpt_filename = self.parameters['zpt_weights_file']
        puid_filename = self.parameters['puid_sf_file']
        
        self.extractor = extractor()
        self.extractor.add_weight_sets([f"* * {zpt_filename}"])
        self.extractor.add_weight_sets([f"* * {puid_filename}"])

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
        
        self.puLookup = util.load('data/pileup/puLookup.coffea')
    
    @property
    def accumulator(self):
        return self._accumulator
    
    @property
    def columns(self):
        return self._columns
    
    def process(self, df):
        # TODO: generate lepton SF for 2017/2018
        # TODO: PU reweighting for 2017/2018
        # TODO: NNLOPS reweighting (ggH)
        # TODO: btag sf
        # TODO: JEC, JER
        # TODO: filter by event number to make sure BDT/DNN is evaluated only on test events
        # TODO: Add systematic uncertainties
        # TODO: check updated jet PUID for 2018 (caltech slides)
        # TODO: fix FSR photon index
        
        # Variables to add (all need to be saved for unbinned analysis):
        # dimuon_costhetaCS, dimuon_phiCS
        # zeppenfeld
        # min dEta and dPhi between dimuon system and two jets (meaning, argmin is jet number 0 or 1)
        # Rpt, pt-centrality
        # Nsoftjets (2GeV and 5GeV), HTsoft
        # mass, eta, phi of mumujj system
        # dimuon mass resolution (calibrated)
        
        if self.timer:
            self.timer.update()
            
        output = self.accumulator.identity()
        dataset = df.metadata['dataset']
        isData = 'data' in dataset
            
        nEvts = df.shape[0]

        if isData:
            lumi_info = LumiMask(self.parameters['lumimask'])
            lumimask = lumi_info(df.run.flatten(), df.luminosityBlock.flatten())
            event_weight = np.ones(nEvts)
        else:    
            lumimask = np.ones(nEvts, dtype=bool)
            genweight = df.genWeight.flatten()
            pu_weight = self.puLookup(dataset, df.Pileup.nTrueInt)
            event_weight = genweight*pu_weight
            if dataset in self.lumi_weights:
                event_weight = event_weight*self.lumi_weights[dataset]
            if self.parameters["do_l1prefiring_wgts"]:
                event_weight = event_weight*df.L1PreFiringWeight.Nom.flatten()
    
        hlt = np.zeros(nEvts, dtype=bool)
        for hlt_path in self.parameters['hlt']:
            hlt = hlt | df.HLT[hlt_path]
            
        mask = hlt & lumimask
    
        # Filter 0: HLT & lumimask
        #--------------------------------#    
        df = df[mask]
        event_weight = event_weight[mask]
        if self.timer:
            self.timer.add_checkpoint("Applied HLT and lumimask")
        #--------------------------------# 
        
        # FSR recovery
        if self.do_fsr:
            mu = df.Muon[df.Muon.pt > self.parameters["muon_pt_cut"]]
            fsr = df.FsrPhoton
            muons_pt = mu.pt.flatten()
            muons_eta = mu.eta.flatten()
            muons_phi = mu.phi.flatten()
            muons_mass = mu.mass.flatten()
            muons_iso = mu.pfRelIso04_all.flatten()
            muons_fsrIndex = mu.fsrPhotonIdx.flatten()
            muons_offsets = mu.counts2offsets(mu.counts)

            fsr_pt = fsr.pt.flatten()
            fsr_eta = fsr.eta.flatten()
            fsr_phi = fsr.phi.flatten()
            fsr_offsets =fsr.counts2offsets(fsr.counts)

            muons_pt, muons_eta, muons_phi, muons_mass, muons_iso = correct_muon_with_fsr(muons_offsets, fsr_offsets,\
                                  muons_pt, muons_eta, muons_phi,\
                                  muons_mass, muons_iso,\
                                  muons_fsrIndex, fsr_pt, fsr_eta,\
                                  fsr_phi) 

            fsr_updated_attrs = {'pt': muons_pt, 'eta':muons_eta, 'phi':muons_phi, 'mass':muons_mass, 'pfRelIso04_all':muons_iso}

            muonarrays = {key:mu[key].flatten() for key in mu.columns}
            muonarrays.update(fsr_updated_attrs)
            muons = JaggedCandidateArray.candidatesfromcounts(mu.counts, **muonarrays)
            
        else:
            muons = df.Muon[df.Muon.pt > self.parameters["muon_pt_cut"]]
        
        pass_event_flags = np.ones(df.shape[0], dtype=bool)
        for flag in self.parameters["event_flags"]:
            pass_event_flags = pass_event_flags & df.Flag[flag]
        
        pass_muon_flags = np.ones(df.shape[0], dtype=bool)
        for flag in self.parameters["muon_flags"]:
            pass_muon_flags = pass_muon_flags & muons[flag]

        muons = muons[(abs(muons.eta) < self.parameters["muon_eta_cut"]) &\
                        (muons.pfRelIso04_all < self.parameters["muon_iso_cut"]) &\
                        muons[self.parameters["muon_id"]] & pass_muon_flags]            

        two_os_muons = ((muons.counts == 2) & (muons['charge'].prod() == -1))
        
        electrons = df.Electron[(df.Electron.pt > self.parameters["electron_pt_cut"]) &\
                                     (abs(df.Electron.eta) < self.parameters["electron_eta_cut"]) &\
                                     (df.Electron[self.parameters["electron_id"]] == 1)]
                
        electron_veto = (electrons.counts==0)
        good_pv = (df.PV.npvsGood > 0)
            
        event_filter = (pass_event_flags & two_os_muons & electron_veto & good_pv).flatten()
        
        
        # Filter 1: Event selection
        #--------------------------------#    
        df = df[event_filter]
        muons = muons[event_filter]
        event_weight = event_weight[event_filter]
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
        
        if self.do_roccor:
            mu1 = JaggedCandidateArray.candidatesfromcounts(
                np.ones(mu1.pt.shape),
                pt=mu1.pt.flatten()*apply_roccor(self.rochester, isData, mu1).flatten(),
                eta=mu1.eta.flatten(),
                phi=mu1.phi.flatten(),
                mass=mu1.mass.flatten(),
                pfRelIso04_all=mu1.pfRelIso04_all.flatten()
            )
            mu2 = JaggedCandidateArray.candidatesfromcounts(
                np.ones(mu2.pt.shape),
                pt=mu2.pt.flatten()*apply_roccor(self.rochester, isData, mu2).flatten(),
                eta=mu2.eta.flatten(),
                phi=mu2.phi.flatten(),
                mass=mu2.mass.flatten(),
                pfRelIso04_all=mu2.pfRelIso04_all.flatten()
            )
            if self.timer:
                self.timer.add_checkpoint("Applied Rochester")


        if 'dy' in dataset:
            zpt_weights = self.evaluator[self.zpt_path](dimuons.pt).flatten()
            event_weight = event_weight*zpt_weights

        mu_pass_leading_pt = muons[(muons.pt > self.parameters["muon_leading_pt"]) &\
                                   (muons.pfRelIso04_all < self.parameters["muon_trigmatch_iso"]) &\
                                   muons[self.parameters["muon_trigmatch_id"]]]
        
        
        if self.do_fsr:
            trigmuarrays = {key:df.TrigObj[key].flatten() for key in df.TrigObj.columns}
            trigmuarrays.update({'mass':0})
            trig_muons = JaggedCandidateArray.candidatesfromcounts(df.TrigObj.counts, **trigmuarrays)

            trig_muons = trig_muons[trig_muons.id == 13]
            muTrig = mu_pass_leading_pt.cross(trig_muons, nested = True)
            _,_,dr = delta_r(muTrig.i0.eta, muTrig.i1.eta, muTrig.i0.phi, muTrig.i1.phi)
            matched = (dr < self.parameters["muon_trigmatch_dr"])
        else:
            trig_muons = df.TrigObj[df.TrigObj.id == 13]        
            muTrig = mu_pass_leading_pt.cross(trig_muons, nested = True)
            matched = (muTrig.i0.delta_r(muTrig.i1) < self.parameters["muon_trigmatch_dr"])

        # at least one muon matched with L3 object, and that muon passes pt, iso and id cuts
        trig_matched = (mu_pass_leading_pt[matched.any()].counts>0)

        
        dimuon_filter = ((mu1.pt>self.parameters["muon_leading_pt"]) &\
                         trig_matched &\
                         (dimuons.mass > self.mass_window[0]) & (dimuons.mass < self.mass_window[1])).flatten()
        if not isData:
            muID = self.mu_id_sf(muons.eta.compact(), muons.pt.compact())
            muIso = self.mu_iso_sf(muons.eta.compact(), muons.pt.compact())
            muTrig = self.mu_iso_sf(abs(muons.eta.compact()), muons.pt.compact())
            muSF = (muID*muIso*muTrig).prod()
            event_weight = event_weight*muSF
#             muIDerr = self.mu_id_err(muons.eta, muons.pt)
#             muIsoerr = self.mu_iso_err(muons.eta, muons.pt)
#             muTrigerr = self.mu_iso_err(abs(muons.eta), muons.pt)
#             muSF_up = ((muID + muIDerr) * (muIso + muIsoerr) * (muTrig + muTrigerr)).prod()
#             muSF_down = ((muID - muIDerr) * (muIso - muIsoerr) * (muTrig - muTrigerr)).prod() 

    
        # Filter 2: Dimuon pair selection
        #--------------------------------#
        df = df[dimuon_filter]   
        mu1 = mu1[dimuon_filter] 
        mu2 = mu2[dimuon_filter] 
        muons = muons[dimuon_filter]
        dimuons = dimuons[dimuon_filter]
        event_weight = event_weight[dimuon_filter]
        if self.timer:
            self.timer.add_checkpoint("Applied dimuon cuts")
        #--------------------------------#   

                 
        if self.do_fsr:
            
            jetarrays = {key:df.Jet[key].flatten() for key in df.Jet.columns}        
            jet = JaggedCandidateArray.candidatesfromcounts(df.Jet.counts, **jetarrays)

            mujet = jet.cross(muons, nested=True)
            _,_,deltar_mujet = delta_r(mujet.i0.eta, mujet.i1.eta, mujet.i0.phi, mujet.i1.phi)
        else:
            mujet = df.Jet.cross(muons, nested=True)
            deltar_mujet = mujet.i0.delta_r(mujet.i1)
        deltar_mujet_ok =  (deltar_mujet > self.parameters["min_dr_mu_jet"]).all()
        
        # Jet ID
        if "loose" in self.parameters["jet_id"]:
            jet_id = (df.Jet.jetId >= 1)
        elif "tight" in self.parameters["jet_id"]:
            if '2016' in self.year:
                jet_id = (df.Jet.jetId >= 3)
            else:
                jet_id = (df.Jet.jetId >= 2)
        else:
            jet_id = df.Jet.ones_like()

        # Jet PU ID
        jet_puid_opt = self.parameters["jet_puid"]
        jet_puid_wps = {
            "loose": (((df.Jet.puId >= 4) & (df.Jet.pt < 50)) | (df.Jet.pt > 50)),
            "medium": (((df.Jet.puId >= 4) & (df.Jet.pt < 50)) | (df.Jet.pt > 50)),
            "tight": (((df.Jet.puId >= 4) & (df.Jet.pt < 50)) | (df.Jet.pt > 50)),
        }
        if jet_puid_opt in ["loose", "medium", "tight"]:
            jet_puid = jet_puid_wps[jet_puid_opt]
        elif "2017corrected" in jet_puid_opt:
            eta_window = (abs(df.Jet.eta)>2.6)&(abs(df.Jet.eta)<3.0)
            jet_puid = (eta_window & jet_puid_wps['tight']) | (~eta_window & jet_puid_wps['loose'])
            jet_puid_opt = "loose" # for sf evaluation
        else:
            jet_puid = df.Jet.ones_like()
          
        # Jet selection
        jet_selection = ((df.Jet.pt > self.parameters["jet_pt_cut"]) &\
                         (abs(df.Jet.eta) < self.parameters["jet_eta_cut"]) &\
                         jet_id & jet_puid & (df.Jet.qgl > -2) & deltar_mujet_ok)

        jet_puid = jet_puid[jet_selection]
        jets = df.Jet[jet_selection]

        # Jet PUID scale factors
        wp_dict = {"loose": "L", "medium": "M", "tight": "T"}
        wp = wp_dict[jet_puid_opt]
        h_eff_name = f"h2_eff_mc{self.year}_{wp}"
        h_sf_name = f"h2_eff_sf{self.year}_{wp}"
        puid_eff = self.evaluator[h_eff_name](jets.pt, jets.eta)
        puid_sf = self.evaluator[h_sf_name](jets.pt, jets.eta)
        jets_passed = (jets.pt>25) & (jets.pt<50) & jet_puid
        jets_failed = (jets.pt>25) & (jets.pt<50) & (~jet_puid)
        
        pMC   = puid_eff[jets_passed].prod() * (1.-puid_eff[jets_failed]).prod() 
        pData = puid_eff[jets_passed].prod()*puid_sf[jets_passed].prod() *\
                (1.-puid_eff[jets_failed].prod()*puid_sf[jets_failed]).prod()
            
        puid_weight = np.ones(len(event_weight))
        puid_weight[jet_selection.any()] = np.divide(pData[jet_selection.any()], pMC[jet_selection.any()])
        if not isData:
            event_weight = event_weight * puid_weight
        
        # Separate from ttH and VH phase space        
        nBtagLoose = jets[(jets.btagDeepB>self.parameters["btag_loose_wp"]) & (abs(jets.eta)<2.5)].counts
        nBtagMedium = jets[(jets.btagDeepB>self.parameters["btag_medium_wp"])  & (abs(jets.eta)<2.5)].counts

        jet_filter = ((nBtagLoose<2)&(nBtagMedium<1))

        # Filter 3: Jet filter
        #--------------------------------#
        df = df[jet_filter]   
        mu1 = mu1[jet_filter] 
        mu2 = mu2[jet_filter] 
        dimuons = dimuons[jet_filter]
        jets = jets[jet_filter]
        jet_selection = jet_selection[jet_filter]
        event_weight = event_weight[jet_filter]
        if self.timer:
            self.timer.add_checkpoint("Applied jet cuts")

        #--------------------------------#

        
        # In the computations below I'm trying to keep the size of objects in first dimension the same
        # as it is in the previous step, in order to be able to apply event_weight similarly for all variables
        
        one_jet = (jet_selection.any() & (jets.counts>0))
        two_jets = (jet_selection.any() & (jets.counts>1))
        
        category = np.empty(df.shape[0], dtype=object)
        category[~two_jets] = 'ggh_01j'
        
        jet1_mask = np.zeros(len(event_weight))
        jet1_mask[one_jet] = 1
        jet1 = JaggedCandidateArray.candidatesfromcounts(
            jet1_mask,
            pt=jets[one_jet,0].pt.flatten(),
            eta=jets[one_jet,0].eta.flatten(),
            phi=jets[one_jet,0].phi.flatten(),
            mass=jets[one_jet,0].mass.flatten(),
            qgl=jets[one_jet,0].qgl.flatten()
        )
        
        jet2_mask = np.zeros(len(event_weight))
        jet2_mask[two_jets] = 1
        jet2 = JaggedCandidateArray.candidatesfromcounts(
            jet2_mask,
            pt=jets[two_jets,1].pt.flatten(),
            eta=jets[two_jets,1].eta.flatten(),
            phi=jets[two_jets,1].phi.flatten(),
            mass=jets[two_jets,1].mass.flatten(),
            qgl=jets[two_jets,1].qgl.flatten()
        )
        
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

        vbf_cut = (dijet.mass>400)&(dijet_deta>2.5)
        category[two_jets&(~vbf_cut)] = 'ggh_2j'
        category[two_jets&vbf_cut] = 'vbf'
        
        if self.timer:
            self.timer.add_checkpoint("Computed jet variables")
        
        assert(np.count_nonzero(category)==category.shape[0]), "Not all events have been categorized!"
        channels = list(set(category))
        
#----------------------------------------------------------------------------#
        # flatten variables where exactly one value per event expected
#        deta_mumu = abs(mu1.eta - mu2.eta).flatten()
#        dphi_mumu = abs(mu1.p4.delta_phi(mu2.p4)).flatten()
#        dr_mumu = np.sqrt(deta_mumu*deta_mumu + dphi_mumu*dphi_mumu)
        deta_mumu, dphi_mumu, dr_mumu = delta_r(mu1.eta.flatten(), mu2.eta.flatten(), mu1.phi.flatten(), mu2.phi.flatten())
        deta_mumuj1 = np.zeros(one_jet.shape[0])
        dphi_mumuj1 = np.zeros(one_jet.shape[0])
        dr_mumuj1 = np.zeros(one_jet.shape[0])
        
        deta_mumuj1[one_jet], dphi_mumuj1[one_jet], dr_mumuj1[one_jet] = delta_r(dimuons[one_jet].eta.flatten(),\
                                                                              jet1.eta.flatten(),\
                                                                              dimuons[one_jet].phi.flatten(),\
                                                                              jet1.phi.flatten())
        deta_mumuj2 = np.zeros(two_jets.shape[0])
        dphi_mumuj2 = np.zeros(two_jets.shape[0])
        dr_mumuj2 = np.zeros(two_jets.shape[0])
        
        deta_mumuj2[two_jets], dphi_mumuj2[two_jets], dr_mumuj2[two_jets] = delta_r(dimuons[two_jets].eta.flatten(),\
                                                                              jet2.eta.flatten(),\
                                                                              dimuons[two_jets].phi.flatten(),\
                                                                              jet2.phi.flatten())

        
        variable_map = {
            'dimuon_mass': dimuons.mass.flatten(),
            'dimuon_pt': dimuons.pt.flatten(),
            'dimuon_eta': dimuons.eta.flatten(),
            'dimuon_phi': dimuons.phi.flatten(),
            'dimuon_dEta': deta_mumu,
            'dimuon_dPhi': dphi_mumu,
            'dimuon_dR': dr_mumu,
            
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
     
            'jj_mass': dijet.mass,
            'jj_pt': dijet.pt,
            'jj_eta': dijet.eta,
            'jj_phi': dijet.phi,
            'jj_dEta': dijet_deta,
            'jj_dPhi': dijet_dphi,      
 
            'njets': jets.counts.flatten(),
        
            'npv': df.PV.npvsGood.flatten(),
            'met': df.MET.pt.flatten(),
            'event': df.event.flatten(),
            'event_weight': event_weight,
        }

        # Evaluate DNN 

        if self.evaluate_dnn:                
            training_features = ['dimuon_mass', 'dimuon_pt', 'dimuon_eta', 'dimuon_dEta', 'mu1_pt', 'mu2_pt']
            n_rows = len(dimuons.mass.flatten())
            df_for_dnn = pd.DataFrame(columns=training_features)
                
            for tf in training_features:
                feature_column = variable_map[tf]
                assert(n_rows==len(feature_column))
                df_for_dnn[tf] = feature_column
                
            from keras.models import load_model
            import keras.backend as K
            import tensorflow as tf
            config = tf.ConfigProto()
            config.intra_op_parallelism_threads=1
            config.inter_op_parallelism_threads=1
            K.set_session(tf.Session(config=config))

            # BOTTLENECK: can't load model outside of a worker
            # https://github.com/keras-team/keras/issues/9964
            dnn_model = load_model('output/trained_models/test.h5')
            
            dnn_score = dnn_model.predict(df_for_dnn[training_features]).flatten()
            variable_map['dnn_score'] = dnn_score
            
            if self.timer:
                self.timer.add_checkpoint("Evaluated DNN")


        #################### Fill outputs ####################
        #------------------ Binned outputs ------------------#  
        for vname, expression in variable_map.items():
            regions = get_regions(variable_map['dimuon_mass'])            
            for cname in channels:
                ccut = (category==cname)
                for rname, rcut in regions.items():
                    if (dataset in self.overlapping_samples) and (dataset not in self.specific_samples[rname][cname]): continue
                    if isData and ('h-peak' in rname) and ('dimuon_mass' in vname): continue # blinding
                    value = expression[rcut & ccut]
                    if not value.size: continue # skip empty arrays
                    if isinstance(value, awkward.JaggedArray):
                        # correctly fill arrays with empty elements (for example events with 0 jets)
                        weight = event_weight[rcut & ccut][value.any()]
                    else:
                        weight = event_weight[rcut & ccut]
                    output[vname].fill(**{'dataset': dataset, 'region': rname, 'channel': cname,\
                                         vname: value.flatten(), 'weight': weight})

        output['cutflow']['all events'] += nEvts
    

        
        #----------------- Unbinned outputs -----------------#
        
        if dataset in self.datasets_to_save_unbin: # don't need unbinned data for all samples
            for v in self.vars_unbin:
                if v not in variable_map: continue
                for cname in channels:
                    ccut = (category==cname)
                    for rname, rcut in regions.items():
                        if (dataset in self.overlapping_samples) and (dataset not in self.specific_samples[rname][cname]): continue
                        if 'z-peak' in rname: continue # don't need unbinned data under Z-peak
                        output[f'{v}_unbin_{dataset}_c_{cname}_r_{rname}'] += processor.column_accumulator(variable_map[v][rcut & ccut].flatten())
    
        if self.timer:
            self.timer.add_checkpoint("Filled outputs")
            
        if self.timer:
            self.timer.summary()

        return output
    
    def postprocess(self, accumulator):
        return accumulator