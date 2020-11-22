import copy
import awkward
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)
import pandas as pd

from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from coffea.lookup_tools import extractor
from coffea.lookup_tools import txt_converters, rochester_lookup
from coffea.jetmet_tools import FactorizedJetCorrector
from coffea.jetmet_tools import JetCorrectionUncertainty
from coffea.jetmet_tools import JetTransformer
from coffea.jetmet_tools import JetResolution
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.btag_tools import BTagScaleFactor
from coffea.lumi_tools import LumiMask

from python.utils import p4_sum, p4_sum_alt, delta_r, rapidity, cs_variables
from python.timer import Timer
from python.weights import Weights
from python.corrections import musf_lookup, musf_evaluator, pu_lookup
from python.corrections import pu_evaluator, NNLOPS_Evaluator
from python.corrections import roccor_evaluator, get_jec_unc
from python.corrections import qgl_weights, puid_weights, btag_weights
from python.corrections import geofit_evaluator, fsr_evaluator
from python.stxs_uncert import vbf_uncert_stage_1_1, stxs_lookups
from python.mass_resolution import mass_resolution_purdue, mass_resolution_pisa

from config.parameters import parameters
from config.variables import variables


class DimuonProcessor(processor.ProcessorABC):
    def __init__(self, samp_info, do_timer=False, save_unbin=True,
                 do_jecunc=False, do_jerunc=False, do_pdf=True,
                 do_btag_syst=True, auto_pu=True, debug=False,
                 pt_variations=['nominal']):
        if not samp_info:
            print("Samples info missing!")
            return
        self.auto_pu = auto_pu
        self.samp_info = samp_info
        self.year = self.samp_info.year
        self.debug = debug
        self.save_unbin = save_unbin
        self.pt_variations = pt_variations
        self.do_pdf = do_pdf
        self.do_btag_syst = do_btag_syst
        self.parameters = {
            k: v[self.year] for k, v in parameters.items()}

        self.timer = Timer('global') if do_timer else None

        self._columns = self.parameters["proc_columns"]

        self.regions = self.samp_info.regions
        self.channels = self.samp_info.channels

        self.lumi_weights = self.samp_info.lumi_weights

        self.sths_names = ["Yield", "PTH200", "Mjj60", "Mjj120",
                           "Mjj350", "Mjj700", "Mjj1000", "Mjj1500",
                           "PTH25", "JET01"]

        if self.do_btag_syst:
            self.btag_systs = ["jes", "lf", "hfstats1", "hfstats2",
                               "cferr1", "cferr2", "hf", "lfstats1",
                               "lfstats2"]
        else:
            self.btag_systs = []

        self.vars_to_save = set([v.name for v in variables])

        # Prepare lookups for corrections
        rochester_data = txt_converters.convert_rochester_file(
            self.parameters["roccor_file"], loaduncs=True)
        self.roccor_lookup = rochester_lookup.rochester_lookup(
            rochester_data)
        self.musf_lookup = musf_lookup(self.parameters)
        self.pu_lookup = pu_lookup(self.parameters)
        self.pu_lookup_up = pu_lookup(self.parameters, 'up')
        self.pu_lookup_down = pu_lookup(self.parameters, 'down')

        self.btag_lookup = BTagScaleFactor(
            self.parameters["btag_sf_csv"],
            BTagScaleFactor.RESHAPE,
            'iterativefit,iterativefit,iterativefit')
        self.stxs_acc_lookups, self.powheg_xsec_lookup = stxs_lookups()

        # Prepare evaluator for corrections that can be loaded together
        zpt_filename = self.parameters['zpt_weights_file']
        puid_filename = self.parameters['puid_sf_file']

        self.extractor = extractor()
        self.extractor.add_weight_sets([f"* * {zpt_filename}"])
        self.extractor.add_weight_sets([f"* * {puid_filename}"])
        self.extractor.add_weight_sets(
            ["* * data/mass_res_pisa/muonresolution.root"])

        for mode in ["Data", "MC"]:
            label = f"res_calib_{mode}_{self.year}"
            path = self.parameters['res_calib_path']
            file_path = f"{path}/{label}.root"
            self.extractor.add_weight_sets(
                [f"{label} {label} {file_path}"])

        self.extractor.finalize()
        self.evaluator = self.extractor.make_evaluator()

        if '2016' in self.year:
            self.zpt_path = 'zpt_weights/2016_value'
        else:
            self.zpt_path = 'zpt_weights/2017_value'
        self.evaluator[self.zpt_path]._axes =\
            self.evaluator[self.zpt_path]._axes[0]

        # Prepare evaluators for JEC, JER and their systematics
        jetext = extractor()
        jetext.add_weight_sets(self.parameters['jec_weight_sets'])
        jetext.finalize()
        Jetevaluator = jetext.make_evaluator()
        JECcorrector = FactorizedJetCorrector(
            **{name: Jetevaluator[name] for name in
               self.parameters['jec_names']})
        JECuncertainties = JetCorrectionUncertainty(
            **{name: Jetevaluator[name] for name in
               self.parameters['junc_names']})
        JER = JetResolution(
            **{name: Jetevaluator[name] for name in
               self.parameters['jer_names']})
        JERsf = JetResolutionScaleFactor(
            **{name: Jetevaluator[name] for name in
               self.parameters['jersf_names']})
        self.Jet_transformer_JER = JetTransformer(
            jec=None, jer=JER, jersf=JERsf)
        self.Jet_transformer = JetTransformer(
            jec=JECcorrector, junc=JECuncertainties)
        self.JECcorrector_Data = {}
        self.Jet_transformer_data = {}
        self.data_runs = list(
            self.parameters['jec_unc_names_data'].keys())
        for run in self.data_runs:
            self.JECcorrector_Data[run] = FactorizedJetCorrector(
                **{name: Jetevaluator[name] for name in
                   self.parameters['jec_names_data'][run]})
            JECuncertainties_Data = JetCorrectionUncertainty(
                **{name: Jetevaluator[name] for name in
                   self.parameters['junc_names_data'][run]})
            self.Jet_transformer_data[run] = JetTransformer(
                jec=self.JECcorrector_Data[run],
                junc=JECuncertainties_Data)

        all_jec_names = [
            name for name in dir(Jetevaluator) if
            self.parameters['jec_unc_sources'] in name]
        self.JECuncertaintySources = JetCorrectionUncertainty(
            **{name: Jetevaluator[name] for name in all_jec_names})
        self.jet_unc_names = list(self.JECuncertaintySources.levels)

        self.do_jecunc = False
        self.do_jerunc = False
        for ptvar in self.pt_variations:
            ptvar_ = ptvar.replace('_up', '').replace('_down', '')
            if ptvar_ in self.parameters["jec_unc_to_consider"]:
                self.do_jecunc = True
            jers = ['jer1', 'jer2', 'jer3', 'jer4', 'jer5', 'jer6']
            if ptvar_ in jers:
                self.do_jerunc = True

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    def process(self, df):
        # ------------------------------------------------------------#
        # Filter out events not passing HLT or having
        # less than 2 muons.
        # ------------------------------------------------------------#
        if self.timer:
            self.timer.update()

        dataset = df.metadata['dataset']

        is_mc = 'data' not in dataset

        # ------------------------------------------------------------#
        # Apply HLT, lumimask, genweights, PU weights
        # and L1 prefiring weights
        # ------------------------------------------------------------#

        numevents = df.shape[0]
        weights = Weights(df)

        if is_mc:
            nTrueInt = df.Pileup.nTrueInt.flatten()
        else:
            nTrueInt = np.zeros(numevents, dtype=bool)
        output = pd.DataFrame()
        output['run'] = df.run.flatten()
        output['event'] = df.event.flatten()
        output['npv'] = df.PV.npvs.flatten()
        output['nTrueInt'] = nTrueInt
        output['met'] = df.MET.pt.flatten()

        if is_mc:
            mask = np.ones(numevents, dtype=bool)

            # --------------------------------------------------------#
            # Apply gen.weights, pileup weights, lumi weights,
            # L1 prefiring weights
            # --------------------------------------------------------#

            genweight = df.genWeight.flatten()
            weights.add_weight('genwgt', genweight)
            if self.auto_pu:
                pu_distribution = df.Pileup.nTrueInt
                self.pu_lookup = pu_lookup(
                    self.parameters, 'nom', auto=pu_distribution)
                self.pu_lookup_up = pu_lookup(
                    self.parameters, 'up', auto=pu_distribution)
                self.pu_lookup_down = pu_lookup(
                    self.parameters, 'down', auto=pu_distribution)
            pu_weight = pu_evaluator(
                self.pu_lookup, numevents, df.Pileup.nTrueInt)
            pu_weight_up = pu_evaluator(
                self.pu_lookup_up, numevents, df.Pileup.nTrueInt)
            pu_weight_down = pu_evaluator(
                self.pu_lookup_down, numevents, df.Pileup.nTrueInt)
            weights.add_weight_with_variations(
                'pu_wgt', pu_weight, pu_weight_up, pu_weight_down)
            weights.add_weight('lumi', self.lumi_weights[dataset])
            if self.parameters["do_l1prefiring_wgts"]:
                weights.add_weight_with_variations(
                    'l1prefiring_wgt',
                    df.L1PreFiringWeight.Nom.flatten(),
                    df.L1PreFiringWeight.Up.flatten(),
                    df.L1PreFiringWeight.Dn.flatten())

        else:
            lumi_info = LumiMask(self.parameters['lumimask'])
            mask = lumi_info(
                df.run.flatten(), df.luminosityBlock.flatten())

        hlt = np.zeros(df.shape[0], dtype=bool)
        for hlt_path in self.parameters['hlt']:
            if hlt_path in df.HLT.columns:
                hlt = hlt | df.HLT[hlt_path]
        mask = mask & hlt

        if self.timer:
            self.timer.add_checkpoint("Applied HLT and lumimask")

        # ------------------------------------------------------------#
        # Update muon kinematics with Rochester correction,
        # FSR recovery and GeoFit correction
        # Raw pT and eta are stored to be used in event selection
        # ------------------------------------------------------------#

        df.Muon['pt_raw'] = df.Muon.pt
        df.Muon['eta_raw'] = df.Muon.eta
        df.Muon['phi_raw'] = df.Muon.phi
        df.Muon['pfRelIso04_all_raw'] = df.Muon.pfRelIso04_all

        roch_corr, roch_err = roccor_evaluator(
            self.roccor_lookup, is_mc, df.Muon)
        df.Muon['pt'] = df.Muon.pt*roch_corr

        if self.timer:
            self.timer.add_checkpoint("Rochester correction")

        # df.Muon['pt_scale_up'] = df.Muon.pt+df.Muon.pt*roch_err
        # df.Muon['pt_scale_down'] = df.Muon.pt-df.Muon.pt*roch_err
        # muons_pts = {'nominal': df.Muon.pt}
        # 'scale_up':df.Muon.pt_scale_up,
        # 'scale_down':df.Muon.pt_scale_down}

        if True:  # reserved for loop over muon pT variations
            # for
            fsr_offsets = awkward.JaggedArray.counts2offsets(
                df.FsrPhoton.counts)
            muons_offsets = awkward.JaggedArray.counts2offsets(
                df.Muon.counts)
            fsr_pt = np.array(df.FsrPhoton.pt.flatten(), dtype=float)
            fsr_eta = np.array(df.FsrPhoton.eta.flatten(), dtype=float)
            fsr_phi = np.array(df.FsrPhoton.phi.flatten(), dtype=float)
            fsr_iso = np.array(
                df.FsrPhoton.relIso03.flatten(), dtype=float)
            fsr_drEt2 = np.array(
                df.FsrPhoton.dROverEt2.flatten(), dtype=float)
            has_fsr = np.zeros(len(df.Muon.pt.flatten()), dtype=bool)
            pt_fsr, eta_fsr, phi_fsr, mass_fsr, iso_fsr, has_fsr =\
                fsr_evaluator(
                    muons_offsets, fsr_offsets,
                    np.array(df.Muon.pt.flatten(), dtype=float),
                    np.array(df.Muon.eta.flatten(), dtype=float),
                    np.array(df.Muon.phi.flatten(), dtype=float),
                    np.array(df.Muon.mass.flatten(), dtype=float),
                    np.array(
                        df.Muon.pfRelIso04_all.flatten(),
                        dtype=float),
                    np.array(
                        df.Muon.fsrPhotonIdx.flatten(),
                        dtype=int),
                    fsr_pt, fsr_eta, fsr_phi,
                    fsr_iso, fsr_drEt2, has_fsr)
            df.Muon['pt'] = awkward.JaggedArray.fromcounts(
                df.Muon.counts, pt_fsr)
            df.Muon['eta'] = awkward.JaggedArray.fromcounts(
                df.Muon.counts, eta_fsr)
            df.Muon['phi'] = awkward.JaggedArray.fromcounts(
                df.Muon.counts, phi_fsr)
            df.Muon['mass'] = awkward.JaggedArray.fromcounts(
                df.Muon.counts, mass_fsr)
            df.Muon['pfRelIso04_all'] = awkward.JaggedArray.fromcounts(
                df.Muon.counts, iso_fsr)
            df.Muon['pt_fsr'] = df.Muon.pt

            if self.timer:
                self.timer.add_checkpoint("FSR recovery")

            # GeoFit correction
            if False:  # 'dxybs' in df.Muon.columns:
                muons_dxybs = df.Muon.dxybs.flatten()
                muons_charge = df.Muon.charge.flatten()
                muons_pt_gf = geofit_evaluator(
                    df.Muon.pt.flatten(),
                    df.Muon.eta.flatten(),
                    muons_dxybs,
                    muons_charge, self.year, ~has_fsr).flatten()
                df.Muon['pt'] = awkward.JaggedArray.fromcounts(
                    df.Muon.counts, muons_pt_gf)

                if self.timer:
                    self.timer.add_checkpoint("GeoFit correction")

            self.muons_all = df.Muon
            muons = df.Muon

            # --------------------------------------------------------#
            # Select muons that pass pT, eta, isolation cuts,
            # muon ID and quality flags
            # Select events with 2 OS muons, no electrons,
            # passing quality cuts and at least one good PV
            # --------------------------------------------------------#

            pass_event_flags = np.ones(numevents, dtype=bool)
            for flag in self.parameters["event_flags"]:
                pass_event_flags = pass_event_flags &\
                    df.Flag[flag].astype(np.bool)

            pass_muon_flags = np.ones(df.shape[0], dtype=bool)
            for flag in self.parameters["muon_flags"]:
                pass_muon_flags = pass_muon_flags &\
                    muons[flag].astype(np.bool)

            muons = muons[
                (muons.pt_raw > self.parameters["muon_pt_cut"]) &
                (abs(muons.eta_raw) <
                 self.parameters["muon_eta_cut"]) &
                (muons.pfRelIso04_all <
                 self.parameters["muon_iso_cut"]) &
                muons[self.parameters["muon_id"]].astype(np.bool) &
                pass_muon_flags]

            self.muons_all = self.muons_all[
                (self.muons_all.pt_fsr >
                 self.parameters["muon_pt_cut"]) &
                (self.muons_all.pfRelIso04_all <
                 self.parameters["muon_iso_cut"]) &
                self.muons_all[
                 self.parameters["muon_id"]].astype(np.bool)]

            two_os_muons = ((muons.counts == 2) &
                            (muons['charge'].prod() == -1))

            electrons = df.Electron[
                (df.Electron.pt > self.parameters["electron_pt_cut"]) &
                (abs(df.Electron.eta) <
                 self.parameters["electron_eta_cut"]) &
                (df.Electron[self.parameters["electron_id"]] == 1)]

            electron_veto = (electrons.counts == 0)
            good_pv = (df.PV.npvsGood > 0)

            event_filter = (pass_event_flags & two_os_muons &
                            electron_veto & good_pv).flatten()

            mask = mask & event_filter & hlt

            if self.timer:
                self.timer.add_checkpoint("Selected events and muons")

            # --------------------------------------------------------#
            # Initialize muon variables
            # --------------------------------------------------------#

            mu1 = muons[muons.pt.argmax()]
            mu2 = muons[muons.pt.argmin()]

            mu1_variable_names = ['mu1_pt', 'mu1_pt_over_mass',
                                  'mu1_eta', 'mu1_phi', 'mu1_iso']
            mu2_variable_names = ['mu2_pt', 'mu2_pt_over_mass',
                                  'mu2_eta', 'mu2_phi', 'mu2_iso']
            dimuon_variable_names = ['dimuon_mass', 'dimuon_mass_res',
                                     'dimuon_mass_res_rel',
                                     'dimuon_ebe_mass_res',
                                     'dimuon_ebe_mass_res_rel',
                                     'dimuon_pt', 'dimuon_pt_log',
                                     'dimuon_eta', 'dimuon_phi',
                                     'dimuon_dEta', 'dimuon_dPhi',
                                     'dimuon_dR', 'dimuon_rap',
                                     'dimuon_cos_theta_cs',
                                     'dimuon_phi_cs']

            for n in (mu1_variable_names + mu2_variable_names +
                      dimuon_variable_names):
                output[n] = 0.0

            # --------------------------------------------------------#
            # Select events with muons passing leading pT cut
            # and trigger matching
            # --------------------------------------------------------#

            # Events where there is at least one muon passing
            # leading muon pT cut (redundant selection)
            pass_leading_pt = np.zeros(numevents, dtype=bool)
            pass_leading_pt[muons.counts > 0] = (
                mu1.pt_raw > self.parameters["muon_leading_pt"]
            ).flatten()

            # All L3 trigger muons
            df.TrigObj['mass'] = df.TrigObj.zeros_like()
            df.TrigObj = df.TrigObj[(df.TrigObj.id == 13) |
                                    (df.TrigObj.id == -13)]

            # Muons that pass tight id and iso as well as
            # leading muon pT cut
            # mu_for_trigmatch = muons[(
            #     muons.pt_raw > self.parameters["muon_leading_pt"])
            #     &(df.Muon.pfRelIso04_all <
            #     self.parameters["muon_trigmatch_iso"]) &
            #     df.Muon[self.parameters["muon_trigmatch_id"]]
            #    ]

            # For every such muon check if there is
            # a L3 object within dR<0.1
            # muTrig = mu_for_trigmatch.cross(df.TrigObj, nested=True)
            # _, _, dr = delta_r(
            #     muTrig.i0.eta_raw, muTrig.i1.eta,
            #     muTrig.i0.phi, muTrig.i1.phi)
            # has_matched_trigmuon = (
            #     dr < self.parameters["muon_trigmatch_dr"]).any()

            # Events where there is a trigger object matched to
            # a tight-ID tight-Iso muon passing leading pT cut
            # event_passing_trig_match = (mu_for_trigmatch[
            #     has_matched_trigmuon].counts > 0).flatten()

            mask = mask & pass_leading_pt  # & event_passing_trig_match

            if self.timer:
                self.timer.add_checkpoint("Applied trigger matching")

            # --------------------------------------------------------#
            # Fill dimuon and muon variables
            # --------------------------------------------------------#

            two_muons = muons.counts == 2

            mm_pt, mm_eta, mm_phi, mm_mass, mm_rap = p4_sum(
                mu1[two_muons], mu2[two_muons])
            output['dimuon_pt'][two_muons] = mm_pt.flatten()
            output['dimuon_pt_log'] = np.log(output['dimuon_pt'])
            output['dimuon_eta'][two_muons] = mm_eta.flatten()
            output['dimuon_phi'][two_muons] = mm_phi.flatten()
            output['dimuon_mass'][two_muons] = mm_mass.flatten()
            output['dimuon_rap'][two_muons] = mm_rap.flatten()

            mm_deta, mm_dphi, mm_dr = delta_r(
                mu1[two_muons].eta.flatten(),
                mu2[two_muons].eta.flatten(),
                mu1[two_muons].phi.flatten(),
                mu2[two_muons].phi.flatten())

            output['dimuon_dEta'][two_muons] = mm_deta.flatten()
            output['dimuon_dPhi'][two_muons] = mm_dphi.flatten()
            output['dimuon_dR'][two_muons] = mm_dr.flatten()

            output['dimuon_ebe_mass_res'][two_muons] =\
                mass_resolution_purdue(
                    is_mc, self.evaluator, mu1, mu2,
                    output['dimuon_mass'], two_muons, self.year)
            output['dimuon_ebe_mass_res_rel'][two_muons] =\
                output['dimuon_mass_res'][two_muons] /\
                output['dimuon_mass'][two_muons]

            output['dimuon_mass_res_rel'][two_muons] =\
                mass_resolution_pisa(
                    self.extractor, mu1, mu2, two_muons)
            output['dimuon_mass_res'][two_muons] =\
                output['dimuon_mass_res_rel'][two_muons] *\
                output['dimuon_mass'][two_muons]

            cos_theta_cs, phi_cs = cs_variables(mu1, mu2, two_muons)
            output['dimuon_cos_theta_cs'][two_muons] =\
                cos_theta_cs.flatten()
            output['dimuon_phi_cs'][two_muons] = phi_cs.flatten()

            output['mu1_pt'][two_muons] = mu1[two_muons].pt.flatten()
            output['mu1_eta'][two_muons] = mu1[two_muons].eta.flatten()
            output['mu1_phi'][two_muons] = mu1[two_muons].phi.flatten()
            output['mu1_iso'][two_muons] =\
                mu1[two_muons].pfRelIso04_all.flatten()
            output['mu1_pt_over_mass'][two_muons] =\
                np.divide(
                    mu1[two_muons].pt.flatten(),
                    output['dimuon_mass'][two_muons])

            output['mu2_pt'][two_muons] = mu2[two_muons].pt.flatten()
            output['mu2_eta'][two_muons] = mu2[two_muons].eta.flatten()
            output['mu2_phi'][two_muons] = mu2[two_muons].phi.flatten()
            output['mu2_iso'][two_muons] =\
                mu2[two_muons].pfRelIso04_all.flatten()
            output['mu2_pt_over_mass'][two_muons] =\
                np.divide(
                    mu2[two_muons].pt.flatten(),
                    output['dimuon_mass'][two_muons])

            if self.timer:
                self.timer.add_checkpoint("Filled muon variables")

        # ------------------------------------------------------------#
        # Prepare jets
        # ------------------------------------------------------------#
        df.Jet['ptRaw'] = df.Jet.pt * (1 - df.Jet.rawFactor)
        df.Jet['massRaw'] = df.Jet.mass * (1 - df.Jet.rawFactor)
        df.Jet['rho'] = df.fixedGridRhoFastjetAll
        # Alternative way:
        # jet_has_matched_muon = ((df.Jet.matched_muons.pt>15)&
        # (df.Jet.matched_muons.mediumId)&
        # (df.Jet.matched_muons.pfRelIso04_all<0.25)).any()
        jet = df.Jet[df.Jet.pt > 0]
        jet_has_matched_muon = (jet.matched_muons.pt > 0).any()
        jwmm = df.Jet[jet_has_matched_muon].flatten()
        mm = jwmm.matched_muons[:, 0]

        _, _, dr_mujet = delta_r(jwmm.eta, mm.eta, jwmm.phi, mm.phi)

        jet_matched_muon_dr = np.full(len(df.Jet.pt.flatten()), -999.)
        jet_matched_muon_pt = np.full(len(df.Jet.pt.flatten()), -999.)
        jet_matched_muon_iso = np.full(len(df.Jet.pt.flatten()), -999.)
        jet_matched_muon_id = np.full(len(df.Jet.pt.flatten()), -999.)

        jet_matched_muon_dr[jet_has_matched_muon.flatten()] = dr_mujet
        jet_matched_muon_pt[jet_has_matched_muon.flatten()] = mm.pt
        jet_matched_muon_iso[jet_has_matched_muon.flatten()] =\
            mm.pfRelIso04_all
        jet_matched_muon_id[jet_has_matched_muon.flatten()] =\
            mm.mediumId

        df.Jet['has_matched_muon'] = jet_has_matched_muon
        df.Jet['matched_muon_dr'] = awkward.JaggedArray.fromcounts(
            df.Jet.counts, jet_matched_muon_dr)
        df.Jet['matched_muon_pt'] = awkward.JaggedArray.fromcounts(
            df.Jet.counts, jet_matched_muon_pt)
        df.Jet['matched_muon_iso'] = awkward.JaggedArray.fromcounts(
            df.Jet.counts, jet_matched_muon_iso)
        df.Jet['matched_muon_id'] = awkward.JaggedArray.fromcounts(
            df.Jet.counts, jet_matched_muon_id)
        if is_mc:
            gjj = df.Jet.cross(df.GenJet, nested=True)
            _, _, deltar_gjj = delta_r(
                gjj.i0.eta, gjj.i1.eta, gjj.i0.phi, gjj.i1.phi)
            matched_jets = gjj[
                (deltar_gjj == deltar_gjj.min()) &
                (deltar_gjj < 0.4)].i1
            matched_jets_flat = matched_jets.flatten()[
                matched_jets.flatten().counts > 0, 0]
            matched_jets_new = awkward.JaggedArray.fromcounts(
                (matched_jets.flatten().counts > 0).astype(int),
                matched_jets_flat)
            df.Jet['matched_genjet'] = awkward.JaggedArray.fromcounts(
                matched_jets.counts, matched_jets_new)

        if self.timer:
            self.timer.add_checkpoint("Prepared jets")

        # ------------------------------------------------------------#
        # Apply JEC, get JEC variations
        # ------------------------------------------------------------#

        # self.do_jec = True
        self.do_jec = False
        if ('data' in dataset) and ('2018' in self.year):
            self.do_jec = True
        jet_variation_names = ['nominal'] if\
            ('nominal' in self.pt_variations) else []

        if self.do_jec or self.do_jecunc:
            if is_mc:
                if self.do_jec:
                    jets = JaggedCandidateArray.candidatesfromcounts(
                        df.Jet.counts,
                        **{c: df.Jet[c].flatten() for c in
                           df.Jet.columns if 'matched' not in c})
                    if self.timer:
                        self.timer.add_checkpoint(
                            "Converted jets to JCA")
                    self.Jet_transformer.transform(
                        jets, forceStochastic=False)
                    df.Jet['pt'] = jets.pt
                    df.Jet['eta'] = jets.eta
                    df.Jet['phi'] = jets.phi
                    df.Jet['mass'] = jets.mass
                if self.do_jecunc:
                    for junc_name in self.jet_unc_names:
                        juncs = self.parameters["jec_unc_to_consider"]
                        if junc_name not in juncs:
                            continue
                        up_ = (f"{junc_name}_up" not in
                               self.pt_variations)
                        dn_ = (f"{junc_name}_down" not in
                               self.pt_variations)
                        if up_ and dn_:
                            continue
                        jec_up_down = get_jec_unc(
                            junc_name, df.Jet.pt,
                            df.Jet.eta, self.JECuncertaintySources)
                        jec_corr_up, jec_corr_down =\
                            jec_up_down[:, :, 0], jec_up_down[:, :, 1]
                        pt_name_up = f"pt_{junc_name}_up"
                        pt_name_down = f"pt_{junc_name}_down"
                        df.Jet[pt_name_up] = df.Jet.pt * jec_corr_up
                        df.Jet[pt_name_down] =\
                            df.Jet.pt * jec_corr_down
                        if (f"{junc_name}_up" in self.pt_variations):
                            jet_variation_names += [f"{junc_name}_up"]
                        if (f"{junc_name}_down" in self.pt_variations):
                            jet_variation_names +=\
                                [f"{junc_name}_down"]

            elif self.do_jec:
                for run in self.data_runs:
                    # 'A', 'B', 'C', 'D', etc...
                    if run in dataset:
                        # dataset name is something like 'data_B'
                        jets =\
                            JaggedCandidateArray.candidatesfromcounts(
                                df.Jet.counts,
                                **{c: df.Jet[c].flatten() for c in
                                   df.Jet.columns
                                   if 'matched' not in c})
                        if self.timer:
                            self.timer.add_checkpoint(
                                "Converted jets to JCA")

                        self.Jet_transformer_data[run].transform(
                                jets, forceStochastic=False)
                        df.Jet['pt'] = jets.pt
                        df.Jet['eta'] = jets.eta
                        df.Jet['phi'] = jets.phi
                        df.Jet['mass'] = jets.mass
            if self.timer:
                self.timer.add_checkpoint("Applied JEC")
        if is_mc and self.do_jerunc:
            # for c in df.Jet.columns:
            #     print(c, len(df.Jet[c].flatten()))
            jetarrays = {c: df.Jet[c].flatten() for c in
                         df.Jet.columns if 'matched' not in c}
            pt_gen_jet = df.Jet['matched_genjet'].pt.flatten(axis=0)
            # pt_gen_jet = df.Jet.matched_genjet.pt.flatten(axis=0)
            pt_gen_jet = np.zeros(len(df.Jet.flatten()))
            pt_gen_jet[df.Jet.matched_genjet.pt.flatten(axis=0).counts >
                       0] = df.Jet.matched_genjet.pt.flatten().flatten()
            pt_gen_jet[df.Jet.matched_genjet.pt.flatten(
                axis=0).counts <= 0] = 0
            jetarrays['ptGenJet'] = pt_gen_jet
            jets = JaggedCandidateArray.candidatesfromcounts(
                df.Jet.counts, **jetarrays)
            jet_pt_jec = df.Jet.pt
            self.Jet_transformer_JER.transform(
                jets, forceStochastic=False)
            jet_pt_jec_jer = jets.pt
            jet_pt_gen = jets.ptGenJet
            jer_sf = ((jet_pt_jec_jer - jet_pt_gen) /
                      (jet_pt_jec - jet_pt_gen +
                       (jet_pt_jec == jet_pt_gen) *
                       (jet_pt_jec_jer - jet_pt_jec)))
            jer_down_sf = ((jets.pt_jer_down - jet_pt_gen) /
                           (jet_pt_jec - jet_pt_gen +
                           (jet_pt_jec == jet_pt_gen) * 10.))
            jet_pt_jer_down = jet_pt_gen +\
                (jet_pt_jec - jet_pt_gen) *\
                (jer_down_sf / jer_sf)
            jer_categories = {
                'jer1': (abs(jets.eta) < 1.93),
                'jer2': (abs(jets.eta) > 1.93) & (abs(jets.eta) < 2.5),
                'jer3': ((abs(jets.eta) > 2.5) &
                         (abs(jets.eta) < 3.139) &
                         (jets.pt < 50)),
                'jer4': ((abs(jets.eta) > 2.5) &
                         (abs(jets.eta) < 3.139) &
                         (jets.pt > 50)),
                'jer5': (abs(jets.eta) > 3.139) & (jets.pt < 50),
                'jer6': (abs(jets.eta) > 3.139) & (jets.pt > 50),
            }
            for jer_unc_name, jer_cut in jer_categories.items():
                jer_cut = jer_cut & (jets.ptGenJet > 0)
                up_ = (f"{jer_unc_name}_up" not in self.pt_variations)
                dn_ = (f"{jer_unc_name}_down" not in
                       self.pt_variations)
                if up_ and dn_:
                    continue
                pt_name_up = f"pt_{jer_unc_name}_up"
                pt_name_down = f"pt_{jer_unc_name}_down"
                df.Jet[pt_name_up] = jet_pt_jec
                df.Jet[pt_name_down] = jet_pt_jec
                df.Jet[pt_name_up][jer_cut] = jet_pt_jec_jer[jer_cut]
                df.Jet[pt_name_down][jer_cut] =\
                    jet_pt_jer_down[jer_cut]

                if (f"{jer_unc_name}_up" in self.pt_variations):
                    jet_variation_names += [f"{jer_unc_name}_up"]
                if (f"{jer_unc_name}_down" in self.pt_variations):
                    jet_variation_names += [f"{jer_unc_name}_down"]
            if self.timer:
                self.timer.add_checkpoint("Computed JER nuisances")

        df.Jet['pt_nominal'] = df.Jet.pt
        df.Jet = df.Jet[df.Jet.pt.argsort()]

        if self.timer:
            self.timer.add_checkpoint("Applied JEC/JER (if enabled)")

        # ------------------------------------------------------------#
        # Apply jetID
        # ------------------------------------------------------------#

        if "loose" in self.parameters["jet_id"]:
            jet_id = (df.Jet.jetId >= 1)
        elif "tight" in self.parameters["jet_id"]:
            if '2016' in self.year:
                jet_id = (df.Jet.jetId >= 3)
            else:
                jet_id = (df.Jet.jetId >= 2)
        else:
            jet_id = df.Jet.ones_like()

        good_jet_id = jet_id & (df.Jet.qgl > -2)

        df.Jet = df.Jet[good_jet_id]

        # ------------------------------------------------------------#
        # Calculate other event weights
        # ------------------------------------------------------------#

        if is_mc:
            if ('ggh' in dataset):
                nnlops = NNLOPS_Evaluator('data/NNLOPS_reweight.root')
                nnlopsw = np.ones(numevents, dtype=float)
                if 'amc' in dataset:
                    nnlopsw = nnlops.evaluate(
                        df.HTXS.Higgs_pt, df.HTXS.njets30, "mcatnlo")
                elif 'powheg' in dataset:
                    nnlopsw = nnlops.evaluate(
                        df.HTXS.Higgs_pt, df.HTXS.njets30, "powheg")
                weights.add_weight('nnlops', nnlopsw)

            if ('dy' in dataset) and False:  # disable for now
                zpt_weight = np.ones(numevents, dtype=float)
                zpt_weight[two_muons] =\
                    self.evaluator[self.zpt_path](
                        output['dimuon_pt'][two_muons]).flatten()
                weights.add_weight('zpt_wgt', zpt_weight)

            muID, muID_up, muID_down,\
                muIso, muIso_up, muIso_down,\
                muTrig, muTrig_up, muTrig_down = musf_evaluator(
                    self.musf_lookup, self.year, numevents, muons)
            weights.add_weight_with_variations(
                'muID', muID, muID_up, muID_down)
            weights.add_weight_with_variations(
                'muIso', muIso, muIso_up, muIso_down)
            # weights.add_weight_with_variations(
            #     'muTrig', muTrig, muTrig_up, muTrig_down)

            if ('nominal' in self.pt_variations):
                try:
                    if (('dy_m105_160_amc' in dataset) and
                        (('2017' in self.year) or
                         ('2018' in self.year))):
                        lhefactor = 2.
                    else:
                        lhefactor = 1.
                    nLHEScaleWeight = df.LHEScaleWeight.counts

                    lhe_ren_up = np.full(
                        numevents, df.LHEScaleWeight[:, 6] * lhefactor,
                        dtype=float)
                    lhe_ren_up[nLHEScaleWeight > 8] =\
                        df.LHEScaleWeight[
                            nLHEScaleWeight > 8][:, 7] * lhefactor
                    lhe_ren_up[nLHEScaleWeight > 30] =\
                        df.LHEScaleWeight[
                            nLHEScaleWeight > 30][:, 34] * lhefactor
                    lhe_ren_down = np.full(
                        numevents, df.LHEScaleWeight[:, 1] * lhefactor,
                        dtype=float)
                    lhe_ren_down[nLHEScaleWeight > 8] =\
                        df.LHEScaleWeight[
                            nLHEScaleWeight > 8][:, 1] * lhefactor
                    lhe_ren_down[nLHEScaleWeight > 30] =\
                        df.LHEScaleWeight[
                            nLHEScaleWeight > 30][:, 5] * lhefactor
                    weights.add_only_variations(
                        'LHERen', lhe_ren_up, lhe_ren_down)
                    lhe_fac_up = np.full(
                        numevents, df.LHEScaleWeight[:, 4] * lhefactor,
                        dtype=float)
                    lhe_fac_up[nLHEScaleWeight > 8] =\
                        df.LHEScaleWeight[
                            nLHEScaleWeight > 8][:, 5] * lhefactor
                    lhe_fac_up[nLHEScaleWeight > 30] =\
                        df.LHEScaleWeight[
                            nLHEScaleWeight > 30][:, 24] * lhefactor
                    lhe_fac_down = np.full(
                        numevents, df.LHEScaleWeight[:, 3] * lhefactor,
                        dtype=float)
                    lhe_fac_down[nLHEScaleWeight > 8] =\
                        df.LHEScaleWeight[
                            nLHEScaleWeight > 8][:, 3] * lhefactor
                    lhe_fac_down[nLHEScaleWeight > 30] =\
                        df.LHEScaleWeight[
                            nLHEScaleWeight > 30][:, 15] * lhefactor
                    weights.add_only_variations(
                        'LHEFac', lhe_fac_up, lhe_fac_down)
                except Exception:
                    weights.add_only_variations(
                        'LHEFac',
                        np.full(
                            output.shape[0], np.nan, dtype='float64'),
                        np.full(
                            output.shape[0], np.nan, dtype='float64'))
                    weights.add_only_variations(
                        'LHERen',
                        np.full(
                            output.shape[0], np.nan, dtype='float64'),
                        np.full(
                            output.shape[0], np.nan, dtype='float64'))

            do_thu = (('vbf' in dataset) and
                      ('dy' not in dataset) and
                      ('nominal' in self.pt_variations))
            if do_thu:
                pass
                for i, name in enumerate(self.sths_names):
                    wgt_up = vbf_uncert_stage_1_1(
                        i, df.HTXS.stage1_1_fine_cat_pTjet30GeV, 1.,
                        self.stxs_acc_lookups, self.powheg_xsec_lookup)
                    wgt_down = vbf_uncert_stage_1_1(
                        i, df.HTXS.stage1_1_fine_cat_pTjet30GeV, -1.,
                        self.stxs_acc_lookups, self.powheg_xsec_lookup)
                    weights.add_only_variations(
                        "THU_VBF_"+name, wgt_up, wgt_down)

        if self.timer:
            self.timer.add_checkpoint("Computed event weights")

        # ------------------------------------------------------------#
        # Calculate getJetMass
        # ------------------------------------------------------------#

        genJetMass = np.zeros(numevents, dtype=float)
        if is_mc:
            gjets = df.GenJet
            gleptons = df.GenPart[
                (df.GenPart.pdgId == 13) |
                (df.GenPart.pdgId == 11) |
                (df.GenPart.pdgId == 15) |
                (df.GenPart.pdgId == -13) |
                (df.GenPart.pdgId == -11) |
                (df.GenPart.pdgId == -15)]
            gl_pair = gjets.cross(gleptons, nested=True)
            _, _, dr_gl = delta_r(
                gl_pair.i0.eta, gl_pair.i1.eta,
                gl_pair.i0.phi, gl_pair.i1.phi)
            isolated = (dr_gl > 0.3).all()
            gjets = gjets[isolated]
            has_two_jets = gjets.counts > 1
            gjet1 = gjets[has_two_jets, 0]
            gjet2 = gjets[has_two_jets, 1]
            _, _, _, genJetMass[has_two_jets], _ = p4_sum(gjet1, gjet2)

        # ------------------------------------------------------------#
        # Loop over JEC variations and fill jet variables
        # ------------------------------------------------------------#

        output.columns = pd.MultiIndex.from_product(
            [output.columns, ['']], names=['Variable', 'Variation'])
        for v_name in jet_variation_names:
            output = self.jec_loop(
                v_name, is_mc, df, dataset, mask, muons,
                mu1, mu2, two_muons, weights, numevents,
                genJetMass, output)
        if self.timer:
            self.timer.add_checkpoint("Completed JEC loop")

        # ------------------------------------------------------------#
        # PDF variations
        # ------------------------------------------------------------#

        if self.do_pdf and is_mc and ('nominal' in self.pt_variations):
            do_pdf = (("dy" in dataset or
                       "ewk" in dataset or
                       "ggh" in dataset or
                       "vbf" in dataset) and
                      ('mg' not in dataset))
            if do_pdf:
                pdf_wgts = df.LHEPdfWeight[
                    :, 0:self.parameters["n_pdf_variations"]]
                if '2016' in self.year:
                    max_replicas = 0
                    if 'dy' in dataset:
                        max_replicas = 100
                    elif 'ewk' in dataset:
                        max_replicas = 33
                    else:
                        max_replicas = 100
                    for i in range(max_replicas):
                        output[f"pdf_mcreplica{i}"] = pdf_wgts[:, i]
                else:
                    weights.add_only_variations(
                        "pdf_2rms",
                        (1 + 2 * pdf_wgts.std()),
                        (1 - 2 * pdf_wgts.std()))

        # ------------------------------------------------------------#
        # Fill outputs
        # ------------------------------------------------------------#
        mass = output.dimuon_mass
        output['r'] = None
        output.loc[((mass > 76) & (mass < 106)), 'r'] = "z-peak"
        output.loc[((mass > 110) & (mass < 115.03)) |
                   ((mass > 135.03) & (mass < 150)),
                   'r'] = "h-sidebands"
        output.loc[((mass > 115.03) & (mass < 135.03)), 'r'] = "h-peak"
        output['s'] = dataset
        output['year'] = int(self.year)

        for wgt in weights.df.columns:
            if ('up' not in wgt) and ('down' not in wgt):
                continue
            output[f'wgt_{wgt}'] = weights.get_weight(wgt)

        columns_to_save = [c for c in output.columns
                           if (c[0] in self.vars_to_save) or
                           ('wgt_' in c[0]) or ('mcreplica' in c[0]) or
                           (c[0] in ['c', 'r', 's', 'year'])]
        output = output[columns_to_save]
        output = output.reindex(sorted(output.columns), axis=1)

        output = output[
            output.loc[
                :, pd.IndexSlice['c', 'nominal']].isin(self.channels)]
        output = output[output.r.isin(self.regions)]

        if self.timer:
            self.timer.add_checkpoint("Filled outputs")
            self.timer.summary()
        return output

    def jec_loop(self, variation, is_mc, df, dataset, mask, muons,
                 mu1, mu2, two_muons, weights, numevents,
                 genJetMass, output):
        weights = copy.deepcopy(weights)
        # ------------------------------------------------------------#
        # Initialize jet-related variables
        # ------------------------------------------------------------#

        variable_names = ['jet1_pt', 'jet1_eta', 'jet1_rap',
                          'jet1_phi', 'jet1_qgl', 'jet1_id',
                          'jet1_puid', 'jet1_has_matched_muon',
                          'jet1_matched_muon_dr',
                          'jet1_matched_muon_pt',
                          'jet1_matched_muon_iso',
                          'jet1_matched_muon_id',
                          'jet2_pt', 'jet2_eta', 'jet2_rap',
                          'jet2_phi', 'jet2_qgl', 'jet2_id',
                          'jet2_puid', 'jet2_has_matched_muon',
                          'jet2_matched_muon_dr',
                          'jet2_matched_muon_pt',
                          'jet2_matched_muon_iso',
                          'jet2_matched_muon_id',
                          'jj_mass', 'jj_mass_log', 'jj_pt',
                          'jj_eta', 'jj_phi', 'jj_dEta', 'jj_dPhi',
                          'mmj1_dEta', 'mmj1_dPhi', 'mmj1_dR',
                          'mmj2_dEta', 'mmj2_dPhi', 'mmj2_dR',
                          'mmj_min_dEta', 'mmj_min_dPhi', 'mmjj_pt',
                          'mmjj_eta', 'mmjj_phi', 'mmjj_mass', 'rpt',
                          'zeppenfeld', 'll_zstar_log', 'nsoftjets2',
                          'nsoftjets5', 'htsoft2', 'htsoft5']
        variables = {}
        for n in variable_names:
            variables[n] = np.full(numevents, -999.)

        # ------------------------------------------------------------#
        # Select jets for certain pT variation
        # ------------------------------------------------------------#

        if is_mc and variation != 'nominal':
            pt_name = f'pt_{variation}'
        else:
            pt_name = 'pt'

        # Alternative way (doesn't take into account FSR)
        # match_mu = jets.matched_muons
        # deltar_mujet_ok = ((match_mu.pfRelIso04_all>0.25) |
        #                    (~match_mu.mediumId) |
        #                    (match_mu.pt<20)).all().flatten()

        mujet = df.Jet.cross(self.muons_all, nested=True)
        _, _, deltar_mujet = delta_r(
            mujet.i0.eta, mujet.i1.eta_raw, mujet.i0.phi,
            mujet.i1.phi_raw)
        deltar_mujet_ok = (
            deltar_mujet > self.parameters["min_dr_mu_jet"]).all()

        jet_selection = (
            (df.Jet[pt_name] > self.parameters["jet_pt_cut"]) &
            (abs(df.Jet.eta) < self.parameters["jet_eta_cut"])) &\
            deltar_mujet_ok

        # ------------------------------------------------------------#
        # Calculate PUID scale factors and apply PUID
        # ------------------------------------------------------------#

        # Jet PU ID
        jet_puid_opt = self.parameters["jet_puid"]
        puId = df.Jet.puId17 if self.year == "2017" else df.Jet.puId
        jet_puid_wps = {
            "loose": (puId >= 4) | (df.Jet[pt_name] > 50),
            "medium": (puId >= 6) | (df.Jet[pt_name] > 50),
            "tight": (puId >= 7) | (df.Jet[pt_name] > 50),
        }

        if jet_puid_opt in ["loose", "medium", "tight"]:
            jet_puid = jet_puid_wps[jet_puid_opt]
        elif "2017corrected" in jet_puid_opt:
            eta_window = ((abs(df.Jet.eta) > 2.6) &
                          (abs(df.Jet.eta) < 3.0))
            not_eta_window = ((abs(df.Jet.eta) < 2.6) |
                              (abs(df.Jet.eta) > 3.0))
            jet_puid = (eta_window & (puId >= 7)) |\
                       (not_eta_window & jet_puid_wps['loose'])
        else:
            jet_puid = df.Jet.ones_like()

        # Jet PUID scale factors
        if is_mc and False:  # disable for now
            puid_weight = puid_weights(
                self.evaluator, self.year, df.Jet, pt_name,
                jet_puid_opt, jet_puid, numevents)
            weights.add_weight('puid_wgt', puid_weight)

        jet_selection = jet_selection & jet_puid
        df.Jet = df.Jet[jet_selection]

        if self.timer:
            self.timer.add_checkpoint("Selected jets")

        # ------------------------------------------------------------#
        # Fill jet-related variables
        # ------------------------------------------------------------#
        one_jet = (df.Jet.counts > 0)
        two_jets = (df.Jet.counts > 1)

        # jet1_mask = one_jet.astype(int)
        # jet2_mask = two_jets.astype(int)

        jet1 = df.Jet[one_jet][:, 0]
        jet2 = df.Jet[two_jets][:, 1]

        if is_mc:
            qgl_wgt = np.ones(numevents, dtype=float)
            isHerwig = ('herwig' in dataset)
            qgl_wgt[one_jet] = qgl_wgt[one_jet] * qgl_weights(
                jet1, isHerwig)
            qgl_wgt[two_jets] = qgl_wgt[two_jets] * qgl_weights(
                jet2, isHerwig)
            qgl_wgt[one_jet & ~two_jets] = 1.
            qgl_wgt = qgl_wgt / qgl_wgt[mask & two_jets].mean()
            weights.add_weight_with_variations(
                'qgl_wgt', qgl_wgt, up=qgl_wgt * qgl_wgt,
                down=np.ones(numevents, dtype=float))

        # Fill flat arrays of fixed length (numevents)

        variables['jet1_pt'][one_jet] = jet1[pt_name]
        variables['jet1_eta'][one_jet] = jet1.eta
        variables['jet1_rap'][one_jet] = rapidity(jet1)
        variables['jet1_phi'][one_jet] = jet1.phi
        variables['jet1_qgl'][one_jet] = jet1.qgl
        variables['jet1_id'][one_jet] = jet1.jetId
        variables['jet1_puid'][one_jet] = jet1.puId
        variables['jet1_has_matched_muon'][one_jet] =\
            jet1.has_matched_muon
        variables['jet1_matched_muon_dr'][one_jet] =\
            jet1.matched_muon_dr
        variables['jet1_matched_muon_pt'][one_jet] =\
            jet1.matched_muon_pt
        variables['jet1_matched_muon_iso'][one_jet] =\
            jet1.matched_muon_iso
        variables['jet1_matched_muon_id'][one_jet] =\
            jet1.matched_muon_id

        variables['jet2_pt'][two_jets] = jet2[pt_name]
        variables['jet2_eta'][two_jets] = jet2.eta
        variables['jet2_rap'][two_jets] = rapidity(jet2)
        variables['jet2_phi'][two_jets] = jet2.phi
        variables['jet2_qgl'][two_jets] = jet2.qgl
        variables['jet2_id'][two_jets] = jet2.jetId
        variables['jet2_puid'][two_jets] = jet2.puId
        variables['jet2_has_matched_muon'][two_jets] =\
            jet2.has_matched_muon
        variables['jet2_matched_muon_dr'][two_jets] =\
            jet2.matched_muon_dr
        variables['jet2_matched_muon_pt'][two_jets] =\
            jet2.matched_muon_pt
        variables['jet2_matched_muon_iso'][two_jets] =\
            jet2.matched_muon_iso
        variables['jet2_matched_muon_id'][two_jets] =\
            jet2.matched_muon_id

        variables['jj_pt'][two_jets],\
            variables['jj_eta'][two_jets],\
            variables['jj_phi'][two_jets],\
            variables['jj_mass'][two_jets], _ = p4_sum(
                jet1[two_jets[one_jet]], jet2)
        variables['jj_mass_log'] = np.log(variables['jj_mass'])

        variables['jj_dEta'][two_jets],\
            variables['jj_dPhi'][two_jets], _ = delta_r(
                variables['jet1_eta'][two_jets],
                variables['jet2_eta'][two_jets],
                variables['jet1_phi'][two_jets],
                variables['jet2_phi'][two_jets])

        # Definition with rapidity would be different
        variables['zeppenfeld'][two_muons & two_jets] =\
            (output['dimuon_eta'][two_muons & two_jets] -
             0.5 *(variables['jet1_eta'][two_muons & two_jets] +
                   variables['jet2_eta'][two_muons & two_jets]))

        variables['mmjj_pt'][two_muons & two_jets],\
            variables['mmjj_eta'][two_muons & two_jets],\
            variables['mmjj_phi'][two_muons & two_jets],\
            variables['mmjj_mass'][two_muons & two_jets] = p4_sum_alt(
                output['dimuon_pt'][two_muons & two_jets],
                output['dimuon_eta'][two_muons & two_jets],
                output['dimuon_phi'][two_muons & two_jets],
                output['dimuon_mass'][two_muons & two_jets],
                variables['jj_pt'][two_muons & two_jets],
                variables['jj_eta'][two_muons & two_jets],
                variables['jj_phi'][two_muons & two_jets],
                variables['jj_mass'][two_muons & two_jets])

        variables['rpt'][two_muons & two_jets] =\
            variables['mmjj_pt'][two_muons & two_jets] /\
            (output['dimuon_pt'][two_muons & two_jets] +
             variables['jet1_pt'][two_muons & two_jets] +
             variables['jet2_pt'][two_muons & two_jets])
        ll_ystar = np.full(numevents, -999.)
        ll_zstar = np.full(numevents, -999.)

        ll_ystar[two_muons & two_jets] =\
            output['dimuon_rap'][two_muons & two_jets] -\
            (variables['jet1_rap'][two_muons & two_jets] +
             variables['jet2_rap'][two_muons & two_jets]) / 2

        ll_zstar[two_muons & two_jets] = abs(
            ll_ystar[two_muons & two_jets] /
            (variables['jet1_rap'][two_muons & two_jets] -
             variables['jet2_rap'][two_muons & two_jets]))

        variables['ll_zstar_log'][two_muons & two_jets] =\
            np.log(ll_zstar[two_muons & two_jets])

        variables['mmj1_dEta'][two_muons & one_jet],\
            variables['mmj1_dPhi'][two_muons & one_jet],\
            variables['mmj1_dR'][two_muons & one_jet] = delta_r(
                output['dimuon_eta'][two_muons & one_jet],
                variables['jet1_eta'][two_muons & one_jet].flatten(),
                output['dimuon_phi'][two_muons & one_jet],
                variables['jet1_phi'][two_muons & one_jet].flatten())

        variables['mmj2_dEta'][two_muons & two_jets],\
            variables['mmj2_dPhi'][two_muons & two_jets],\
            variables['mmj2_dR'][two_muons & two_jets] = delta_r(
                output['dimuon_eta'][two_muons & two_jets],
                variables['jet2_eta'][two_muons & two_jets].flatten(),
                output['dimuon_phi'][two_muons & two_jets],
                variables['jet2_phi'][two_muons & two_jets].flatten())

        variables['mmj_min_dEta'] = np.where(
            variables['mmj1_dEta'], variables['mmj2_dEta'],
            (variables['mmj1_dEta'] < variables['mmj2_dEta']))

        variables['mmj_min_dPhi'] = np.where(
            variables['mmj1_dPhi'], variables['mmj2_dPhi'],
            (variables['mmj1_dPhi'] < variables['mmj2_dPhi']))

        if self.timer:
            self.timer.add_checkpoint("Filled jet variables")

        # ------------------------------------------------------------#
        # Fill soft activity jet variables
        # ------------------------------------------------------------#

        df.SoftActivityJet['mass'] = df.SoftActivityJet.zeros_like()

        variables['nsoftjets2'],\
            variables['htsoft2'] = self.get_softjet_vars(
                df, df.SoftActivityJet, 2, muons, mu1, mu2,
                jet1, jet2, two_muons, one_jet, two_jets)

        variables['nsoftjets5'],\
            variables['htsoft5'] = self.get_softjet_vars(
                df, df.SoftActivityJet, 5, muons, mu1, mu2,
                jet1, jet2, two_muons, one_jet, two_jets)

        if self.timer:
            self.timer.add_checkpoint("Calculated SA variables")

        # ------------------------------------------------------------#
        # Apply remaining cuts
        # ------------------------------------------------------------#

        leading_jet_pt = np.zeros(numevents, dtype=bool)
        leading_jet_pt[df.Jet.counts > 0] = (
            df.Jet.pt[df.Jet.counts > 0][:, 0] > 35.)
        vbf_cut = ((variables['jj_mass'] > 400) &
                   (variables['jj_dEta'] > 2.5) & leading_jet_pt)

        # ------------------------------------------------------------#
        # Calculate btag SF and apply btag veto
        # ------------------------------------------------------------#
        bjet_sel_mask = mask & two_jets & vbf_cut
        # Btag weight
        btag_wgt = np.ones(numevents)
        if is_mc:
            systs = self.btag_systs if 'nominal' in variation else []
            btag_wgt, btag_syst = btag_weights(
                self, self.btag_lookup, systs, df.Jet,
                weights, bjet_sel_mask, numevents)

            weights.add_weight('btag_wgt', btag_wgt)
            for name, bs in btag_syst.items():
                up = bs[0]
                down = bs[1]
                weights.add_only_variations(
                    f'btag_wgt_{name}', up, down)

        # Separate from ttH and VH phase space
        nBtagLoose = df.Jet[
            (df.Jet.btagDeepB > self.parameters["btag_loose_wp"]) &
            (abs(df.Jet.eta) < 2.5)].counts
        nBtagMedium = df.Jet[
            (df.Jet.btagDeepB > self.parameters["btag_medium_wp"]) &
            (abs(df.Jet.eta) < 2.5)].counts
        mask = mask & (nBtagLoose < 2) & (nBtagMedium < 1)

        # mass = (output['dimuon_mass'] > 115) &\
        #        (output['dimuon_mass'] < 135)

        if self.timer:
            self.timer.add_checkpoint(
                "Applied b-jet SF and b-tag veto")

        # ------------------------------------------------------------#
        # Define categories
        # ------------------------------------------------------------#
        category = np.empty(numevents, dtype=object)
        category[mask & (~two_jets)] = 'ggh_01j'
        category[mask & two_jets & (~vbf_cut)] = 'ggh_2j'
        category[mask & two_jets & vbf_cut] = 'vbf'
        if 'dy' in dataset:
            two_jets_matched = np.zeros(numevents, dtype=bool)
            matched1 =\
                (jet1.matched_genjet.counts > 0)[two_jets[one_jet]]
            matched2 = (jet2.matched_genjet.counts > 0)
            two_jets_matched[two_jets] = matched1 & matched2
            category[mask & two_jets &
                     vbf_cut & (~two_jets_matched)] = 'vbf_01j'
            category[mask & two_jets &
                     vbf_cut & two_jets_matched] = 'vbf_2j'

        # --------------------------------------------------------------#
        # Fill outputs
        # --------------------------------------------------------------#

        variables.update({'njets': df.Jet.counts.flatten()})
        variables.update({'c': category, 'two_jets': two_jets})
        variables.update({'wgt_nominal': weights.get_weight('nominal')})

        # All variables are affected by jet pT because of jet selections:
        # a jet may or may not be selected depending on pT variation,
        # so even variables that don't directly depend on jet pT
        # (e.g. QGL) must be saved for all pT variations.

        for key, val in variables.items():
            output.loc[:, pd.IndexSlice[key, variation]] = val

        return output

    def get_softjet_vars(self, df, softjets, cutoff, muons, mu1, mu2,
                         jet1, jet2, two_muons, one_jet, two_jets):
        nsoftjets = copy.deepcopy(
            df[f'SoftActivityJetNjets{cutoff}'].flatten())
        htsoft = copy.deepcopy(
            df[f'SoftActivityJetHT{cutoff}'].flatten())
        # TODO: sort out the masks (all have different dimensionality)
        mask = two_muons
        mask1j = two_muons & one_jet
        mask1j_ = two_muons[one_jet]
        mask2j = two_muons & two_jets
        mask2j_ = two_muons[two_jets]
        mask2j__ = two_muons[[one_jet]] & two_jets[one_jet]

        j1_two_jets = two_jets[one_jet]

        sj_j = softjets.cross(df.Jet, nested=True)
        sj_mu = softjets.cross(muons, nested=True)

        _, _, dr_sj_j = delta_r(
            sj_j.i0.eta, sj_j.i1.eta, sj_j.i0.phi, sj_j.i1.phi)
        _, _, dr_sj_mu = delta_r(
            sj_mu.i0.eta, sj_mu.i1.eta, sj_mu.i0.phi, sj_mu.i1.phi)

        closest_jet = sj_j[
            (dr_sj_j == dr_sj_j.min()) & (dr_sj_j < 0.4)].i1
        closest_mu = sj_mu[
            (dr_sj_mu == dr_sj_mu.min()) & (dr_sj_mu < 0.4)].i1

        jet1_jagged = awkward.JaggedArray.fromcounts(
            np.ones(len(jet1), dtype=int), jet1)
        jet2_jagged = awkward.JaggedArray.fromcounts(
            np.ones(len(jet2), dtype=int), jet2)

        sj_j1 = closest_jet[one_jet].cross(jet1_jagged, nested=True)
        sj_j2 = closest_jet[two_jets].cross(jet2_jagged, nested=True)
        sj_mu1 = closest_mu.cross(mu1, nested=True)
        sj_mu2 = closest_mu.cross(mu2, nested=True)

        _, _, dr_sj_j1 = delta_r(
            sj_j1.i0.eta, sj_j1.i1.eta, sj_j1.i0.phi, sj_j1.i1.phi)
        _, _, dr_sj_j2 = delta_r(
            sj_j2.i0.eta, sj_j2.i1.eta, sj_j2.i0.phi, sj_j2.i1.phi)
        _, _, dr_sj_mu1 = delta_r(
            sj_mu1.i0.eta, sj_mu1.i1.eta, sj_mu1.i0.phi, sj_mu1.i1.phi)
        _, _, dr_sj_mu2 = delta_r(
            sj_mu2.i0.eta, sj_mu2.i1.eta, sj_mu2.i0.phi, sj_mu2.i1.phi)

        j1match = (dr_sj_j1 < 0.4).any().any()
        j2match = (dr_sj_j2 < 0.4).any().any()
        mumatch = ((dr_sj_mu1 < 0.4).any().any() |
                   (dr_sj_mu2 < 0.4).any().any())

        eta1cut1 = (sj_j1.i0.eta[j1_two_jets] >
                    sj_j1.i1.eta[j1_two_jets]).any().any()
        eta2cut1 = (sj_j2.i0.eta >
                    sj_j2.i1.eta).any().any()
        outer = (eta1cut1[mask2j_]) & (eta2cut1[mask2j_])
        eta1cut2 = (sj_j1.i0.eta[j1_two_jets] <
                    sj_j1.i1.eta[j1_two_jets]).any().any()
        eta2cut2 = (sj_j2.i0.eta <
                    sj_j2.i1.eta).any().any()
        inner = (eta1cut2[mask2j_]) & (eta2cut2[mask2j_])

        nsoftjets[mask] = (
            df[f'SoftActivityJetNjets{cutoff}'][mask] -
            (mumatch[mask] &
             (df.SoftActivityJet.pt > cutoff)[mask]).sum()).flatten()
        nsoftjets[mask1j] = (
            df[f'SoftActivityJetNjets{cutoff}'][mask1j] -
            ((mumatch[mask1j] | j1match[mask1j_]) &
             (df.SoftActivityJet.pt > cutoff)[mask1j]).sum()).flatten()
        nsoftjets[mask2j] = (
            df[f'SoftActivityJetNjets{cutoff}'][mask2j] -
            ((mumatch[mask2j] | j1match[mask2j__] | j2match[mask2j_]) &
             (df.SoftActivityJet.pt > cutoff)[mask2j]).sum()).flatten()

        saj_filter = (mumatch[mask2j] |
                      j1match[mask2j__] |
                      j2match[mask2j_] |
                      outer | inner)
        footprintSAJ = df.SoftActivityJet[mask2j][saj_filter]
        if footprintSAJ.shape[0] > 0:
            htsoft[mask2j] = df[f'SoftActivityJetHT{cutoff}'][mask2j] -\
                        (footprintSAJ.pt *
                         (footprintSAJ.pt > cutoff)).sum()
        return nsoftjets, htsoft

    def postprocess(self, accumulator):
        return accumulator
