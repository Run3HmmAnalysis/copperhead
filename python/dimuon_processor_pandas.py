import copy
import awkward
import awkward as ak
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)
import pandas as pd

import coffea.processor as processor
from coffea.lookup_tools import extractor
from coffea.lookup_tools import txt_converters, rochester_lookup
from coffea.jetmet_tools import CorrectedJetsFactory, JECStack
from coffea.btag_tools import BTagScaleFactor
from coffea.lumi_tools import LumiMask

from python.utils import p4_sum, delta_r, rapidity, cs_variables
from python.timer import Timer
from python.weights import Weights
from python.corrections import musf_lookup, musf_evaluator, pu_lookup
from python.corrections import pu_evaluator, NNLOPS_Evaluator
from python.corrections import qgl_weights  # , puid_weights , btag_weights
from python.corrections import apply_roccor, fsr_recovery, apply_geofit
from python.stxs_uncert import vbf_uncert_stage_1_1, stxs_lookups
from python.mass_resolution import mass_resolution_purdue
# , mass_resolution_pisa

from config.parameters import parameters
from config.variables import variables


class DimuonProcessor(processor.ProcessorABC):
    def __init__(self, samp_info, do_timer=False, save_unbin=True,
                 do_pdf=True, do_btag_syst=True, auto_pu=True,
                 debug=False, pt_variations=['nominal']):
        if not samp_info:
            print("Samples info missing!")
            return
        self.auto_pu = auto_pu
        self.samp_info = samp_info
        self.year = self.samp_info.year
        self.debug = debug
        self.save_unbin = save_unbin
        self.pt_variations = pt_variations
        self.do_roccor = True
        self.do_fsr = True
        self.do_geofit = True
        self.do_nnlops = True
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
            self.parameters["roccor_file"], loaduncs=True
        )
        self.roccor_lookup = rochester_lookup.rochester_lookup(
            rochester_data
        )
        self.musf_lookup = musf_lookup(self.parameters)
        self.pu_lookup = pu_lookup(self.parameters)
        self.pu_lookup_up = pu_lookup(self.parameters, 'up')
        self.pu_lookup_down = pu_lookup(self.parameters, 'down')

        self.btag_lookup = BTagScaleFactor(
            self.parameters["btag_sf_csv"],
            BTagScaleFactor.RESHAPE,
            'iterativefit,iterativefit,iterativefit'
        )
        self.stxs_acc_lookups, self.powheg_xsec_lookup = stxs_lookups()

        # Prepare evaluator for corrections that can be loaded together
        zpt_filename = self.parameters['zpt_weights_file']
        puid_filename = self.parameters['puid_sf_file']

        self.extractor = extractor()
        self.extractor.add_weight_sets([f"* * {zpt_filename}"])
        self.extractor.add_weight_sets([f"* * {puid_filename}"])
        # Doesn't work with uproot4 so far
        # self.extractor.add_weight_sets(
        #     ["* * data/mass_res_pisa/muonresolution.root"])

        for mode in ["Data", "MC"]:
            label = f"res_calib_{mode}_{self.year}"
            path = self.parameters['res_calib_path']
            file_path = f"{path}/{label}.root"
            self.extractor.add_weight_sets(
                [f"{label} {label} {file_path}"]
            )

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
        jetext.add_weight_sets(self.parameters['jec_weight_sets_data'])
        jetext.finalize()
        jet_evaluator = jetext.make_evaluator()

        jec_input_options = {}
        self.jet_factory = {}

        jec_input_options['jec'] = {
            name: jet_evaluator[name]
            for name in self.parameters['jec_stack']
        }
        jec_input_options['junc'] = {
                name: jet_evaluator[name]
                for name in self.parameters['jec_unc_stack']
        }
        for src in self.parameters['jec_unc_sources']:
            for key in jet_evaluator.keys():
                if src in key:
                    jec_input_options['junc'][key] = jet_evaluator[key]
        jec_input_options['jer'] = {
                name: jet_evaluator[name]
                for name in self.parameters['jer_stack']
        }

        for opt in ['jec', 'junc', 'jer']:
            stack = JECStack(jec_input_options[opt])
            name_map = stack.blank_name_map
            name_map['JetPt'] = 'pt'
            name_map['JetMass'] = 'mass'
            name_map['JetEta'] = 'eta'
            name_map['JetA'] = 'area'
            name_map['ptGenJet'] = 'pt_gen'
            name_map['ptRaw'] = 'pt_raw'
            name_map['massRaw'] = 'mass_raw'
            name_map['Rho'] = 'rho'
            self.jet_factory[opt] = CorrectedJetsFactory(
                name_map, stack
            )

        self.data_runs = list(
            self.parameters['junc_sources_data'].keys()
        )
        self.jec_factories_data = {}
        for run in self.data_runs:
            jec_inputs_data = {}
            jec_inputs_data.update(
                {
                    name: jet_evaluator[name] for name
                    in self.parameters['jec_names_data'][run]
                }
            )
            jec_inputs_data.update(
                {
                    name: jet_evaluator[name] for name
                    in self.parameters['junc_names_data'][run]
                }
            )
            for src in self.parameters['junc_sources_data'][run]:
                for key in jet_evaluator.keys():
                    if src in key:
                        jec_inputs_data[key] = jet_evaluator[key]

            jec_stack_data = JECStack(jec_inputs_data)
            name_map = jec_stack_data.blank_name_map
            name_map['JetPt'] = 'pt'
            name_map['JetMass'] = 'mass'
            name_map['JetEta'] = 'eta'
            name_map['JetA'] = 'area'
            name_map['ptGenJet'] = 'pt_gen'
            name_map['ptRaw'] = 'pt_raw'
            name_map['massRaw'] = 'mass_raw'
            name_map['Rho'] = 'rho'
            self.jec_factories_data[run] = CorrectedJetsFactory(
                name_map, jec_stack_data
            )

        self.do_jecunc = False
        self.do_jerunc = False

        # Look at variation names and see if we need to enable
        # calculation of JEC or JER uncertainties
        for ptvar in self.pt_variations:
            ptvar_ = ptvar.replace('_up', '').replace('_down', '')
            if ptvar_ in self.parameters["jec_unc_to_consider"]:
                self.do_jecunc = True
            jers = ['jer1', 'jer2', 'jer3', 'jer4', 'jer5', 'jer6']
            if ptvar_ in jers:
                self.do_jerunc = True

        # If both are enabled, only compute JEC variations,
        # because otherwise processing freezes
        if self.do_jecunc and self.do_jerunc:
            print(
                'Disabling JER variations; currently they cannot be '
                'considered together with JEC variations - eating '
                'too much memory for some reason.\n'
            )
            self.do_jerunc = False

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
            # Initialize timer
            self.timer.update()

        # Dataset name (see definitions in config/datasets.py)
        dataset = df.metadata['dataset']

        is_mc = 'data' not in dataset

        # ------------------------------------------------------------#
        # Apply HLT, lumimask, genweights, PU weights
        # and L1 prefiring weights
        # ------------------------------------------------------------#

        numevents = len(df)

        if is_mc:
            nTrueInt = df.Pileup.nTrueInt
        else:
            nTrueInt = np.zeros(numevents, dtype=bool)

        # All variables that we want to save
        # will be collected into the 'output' dataframe
        output = pd.DataFrame({'run': df.run, 'event': df.event})
        output.index.name = 'entry'
        output['npv'] = df.PV.npvs
        output['nTrueInt'] = nTrueInt
        output['met'] = df.MET.pt

        # Separate dataframe to keep track on weights
        # and their systematic variations
        weights = Weights(output)

        if is_mc:
            mask = np.ones(numevents, dtype=bool)

            # --------------------------------------------------------#
            # Apply gen.weights, pileup weights, lumi weights,
            # L1 prefiring weights
            # --------------------------------------------------------#

            genweight = df.genWeight
            weights.add_weight('genwgt', genweight)
            nTrueInt = np.array(nTrueInt)
            if self.auto_pu:
                self.pu_lookup = pu_lookup(
                    self.parameters, 'nom', auto=nTrueInt
                )
                self.pu_lookup_up = pu_lookup(
                    self.parameters, 'up', auto=nTrueInt
                )
                self.pu_lookup_down = pu_lookup(
                    self.parameters, 'down', auto=nTrueInt
                )
            pu_weight = pu_evaluator(
                self.pu_lookup, numevents, nTrueInt
            )
            pu_weight_up = pu_evaluator(
                self.pu_lookup_up, numevents, nTrueInt
            )
            pu_weight_down = pu_evaluator(
                self.pu_lookup_down, numevents, nTrueInt
            )
            weights.add_weight_with_variations(
                'pu_wgt', pu_weight, pu_weight_up, pu_weight_down
            )
            weights.add_weight('lumi', self.lumi_weights[dataset])
            l1pfw = ak.to_pandas(df.L1PreFiringWeight)
            if self.parameters["do_l1prefiring_wgts"]:
                weights.add_weight_with_variations(
                    'l1prefiring_wgt', l1pfw.Nom, l1pfw.Up, l1pfw.Dn
                )

        else:
            lumi_info = LumiMask(self.parameters['lumimask'])
            mask = lumi_info(df.run, df.luminosityBlock)

        hlt = ak.to_pandas(df.HLT)
        hlt = hlt[self.parameters["hlt"]].sum(axis=1)

        if self.timer:
            self.timer.add_checkpoint("Applied HLT and lumimask")

        # ------------------------------------------------------------#
        # Update muon kinematics with Rochester correction,
        # FSR recovery and GeoFit correction
        # Raw pT and eta are stored to be used in event selection
        # ------------------------------------------------------------#

        # Save raw kinematic variables before computing any corrections
        df['Muon', 'pt_raw'] = df.Muon.pt
        df['Muon', 'eta_raw'] = df.Muon.eta
        df['Muon', 'phi_raw'] = df.Muon.phi
        df['Muon', 'pfRelIso04_all_raw'] = df.Muon.pfRelIso04_all

        # Rochester correction
        if self.do_roccor:
            apply_roccor(df, self.roccor_lookup, is_mc)
            df['Muon', 'pt'] = df.Muon.pt_roch

            if self.timer:
                self.timer.add_checkpoint("Rochester correction")

            # variations will be in branches pt_roch_up and pt_roch_down

            # muons_pts = {
            #     'nominal': df.Muon.pt,
            #     'roch_up':df.Muon.pt_roch_up,
            #     'roch_down':df.Muon.pt_roch_down
            # }

        if True:  # reserved for loop over muon pT variations
            # for

            # FSR recovery
            if self.do_fsr:
                has_fsr = fsr_recovery(df)
                df['Muon', 'pt'] = df.Muon.pt_fsr
                df['Muon', 'eta'] = df.Muon.eta_fsr
                df['Muon', 'phi'] = df.Muon.phi_fsr
                df['Muon', 'pfRelIso04_all'] = df.Muon.iso_fsr

                if self.timer:
                    self.timer.add_checkpoint("FSR recovery")

            df['Muon', 'pt_fsr'] = df.Muon.pt

            # GeoFit correction
            if self.do_geofit and ('dxybs' in df.Muon.fields):
                apply_geofit(df, self.year, ~has_fsr)
                df['Muon', 'pt'] = df.Muon.pt_fsr

                if self.timer:
                    self.timer.add_checkpoint("GeoFit correction")

            muons = ak.to_pandas(df.Muon)

            # --------------------------------------------------------#
            # Select muons that pass pT, eta, isolation cuts,
            # muon ID and quality flags
            # Select events with 2 OS muons, no electrons,
            # passing quality cuts and at least one good PV
            # --------------------------------------------------------#

            flags = ak.to_pandas(df.Flag)
            flags = flags[self.parameters["event_flags"]].product(axis=1)

            muons['pass_flags'] = True
            if self.parameters["muon_flags"]:
                muons['pass_flags'] = muons[
                    self.parameters["muon_flags"]
                ].product(axis=1)

            muons['selection'] = (
                (muons.pt_raw > self.parameters["muon_pt_cut"]) &
                (abs(muons.eta_raw) <
                 self.parameters["muon_eta_cut"]) &
                (muons.pfRelIso04_all <
                 self.parameters["muon_iso_cut"]) &
                muons[self.parameters["muon_id"]] &
                muons.pass_flags
            )

            muons['selection_all'] = (
                (muons.pt_fsr >
                 self.parameters["muon_pt_cut"]) &
                (muons.pfRelIso04_all <
                 self.parameters["muon_iso_cut"]) &
                muons[self.parameters["muon_id"]]
            )

            nmuons = muons[muons.selection].reset_index()\
                .groupby('entry')['subentry'].nunique()
            mm_charge = muons.loc[muons.selection, 'charge']\
                .groupby('entry').prod()

            electrons = df.Electron[
                (df.Electron.pt > self.parameters["electron_pt_cut"]) &
                (abs(df.Electron.eta) <
                 self.parameters["electron_eta_cut"]) &
                (df.Electron[self.parameters["electron_id"]] == 1)
            ]
            electron_veto = ak.to_numpy(ak.count(electrons.pt, axis=1) == 0)

            good_pv = ak.to_pandas(df.PV).npvsGood > 0

            output['two_muons'] = (nmuons == 2)
            output['event_selection'] = (
                (hlt > 0) &
                (flags > 0) &
                (nmuons == 2) &
                (mm_charge == -1) &
                electron_veto &
                good_pv
            )

            if self.timer:
                self.timer.add_checkpoint("Selected events and muons")

            # --------------------------------------------------------#
            # Initialize muon variables
            # --------------------------------------------------------#

            muons = muons[muons.selection & (nmuons == 2)]
            mu1 = muons.loc[muons.pt.groupby('entry').idxmax()]
            mu2 = muons.loc[muons.pt.groupby('entry').idxmin()]
            mu1.index = mu1.index.droplevel('subentry')
            mu2.index = mu2.index.droplevel('subentry')

            mu1_variable_names = [
                'mu1_pt', 'mu1_pt_over_mass',
                'mu1_eta', 'mu1_phi', 'mu1_iso'
            ]
            mu2_variable_names = [
                'mu2_pt', 'mu2_pt_over_mass',
                'mu2_eta', 'mu2_phi', 'mu2_iso'
            ]
            dimuon_variable_names = [
                'dimuon_mass',
                'dimuon_mass_res', 'dimuon_mass_res_rel',
                'dimuon_ebe_mass_res', 'dimuon_ebe_mass_res_rel',
                'dimuon_pt', 'dimuon_pt_log',
                'dimuon_eta', 'dimuon_phi',
                'dimuon_dEta', 'dimuon_dPhi',
                'dimuon_dR', 'dimuon_rap',
                'dimuon_cos_theta_cs', 'dimuon_phi_cs'
            ]
            v_names = (
                mu1_variable_names +
                mu2_variable_names +
                dimuon_variable_names
            )
            for n in (v_names):
                output[n] = 0.0

            # --------------------------------------------------------#
            # Select events with muons passing leading pT cut
            # and trigger matching
            # --------------------------------------------------------#

            # Events where there is at least one muon passing
            # leading muon pT cut
            pass_leading_pt = (
                mu1.pt_raw > self.parameters["muon_leading_pt"]
            )

            output['pass_leading_pt'] = pass_leading_pt
            output['event_selection'] = (
                output.event_selection & output.pass_leading_pt
            )

            if self.timer:
                self.timer.add_checkpoint("Applied trigger matching")

            # --------------------------------------------------------#
            # Fill dimuon and muon variables
            # --------------------------------------------------------#

            # Fill single muon variables
            for v in ['pt', 'ptErr', 'eta', 'phi']:
                output[f'mu1_{v}'] = mu1[v]
                output[f'mu2_{v}'] = mu2[v]

            output['mu1_iso'] = mu1.pfRelIso04_all
            output['mu2_iso'] = mu2.pfRelIso04_all
            output['mu1_pt_over_mass'] = output.mu1_pt / output.dimuon_mass
            output['mu2_pt_over_mass'] = output.mu2_pt / output.dimuon_mass

            # Fill dimuon variables
            mm = p4_sum(mu1, mu2)
            for v in ['pt', 'eta', 'phi', 'mass', 'rap']:
                output[f'dimuon_{v}'] = mm[v]

            output['dimuon_pt_log'] = np.log(output.dimuon_pt)

            mm_deta, mm_dphi, mm_dr = delta_r(
                mu1.eta, mu2.eta,
                mu1.phi, mu2.phi
            )

            output['dimuon_dEta'] = mm_deta
            output['dimuon_dPhi'] = mm_dphi
            output['dimuon_dR'] = mm_dr

            output['dimuon_ebe_mass_res'] = mass_resolution_purdue(
                                                is_mc,
                                                self.evaluator,
                                                output,
                                                self.year
                                            )
            output['dimuon_ebe_mass_res_rel'] = (
                output.dimuon_ebe_mass_res / output.dimuon_mass
            )

            # Doesn't work with uproot4 yet
            # output.dimuon_mass_res_rel =\
            #     mass_resolution_pisa(self.extractor, output)
            #
            # output.dimuon_mass_res =\
            #     output.dimuon_mass_res_rel * output.dimuon_mass

            output['dimuon_cos_theta_cs'],\
                output['dimuon_phi_cs'] = cs_variables(mu1, mu2)

            if self.timer:
                self.timer.add_checkpoint("Filled muon variables")

        # ------------------------------------------------------------#
        # Prepare jets
        # ------------------------------------------------------------#

        df['Jet', 'pt_raw'] = (1 - df.Jet.rawFactor) * df.Jet.pt
        df['Jet', 'mass_raw'] = (1 - df.Jet.rawFactor) * df.Jet.mass
        df['Jet', 'rho'] = ak.broadcast_arrays(
            df.fixedGridRhoFastjetAll, df.Jet.pt
        )[0]

        if is_mc:
            df['Jet', 'pt_gen'] = ak.values_astype(
                ak.fill_none(df.Jet.matched_gen.pt, 0), np.float32
            )

        if self.timer:
            self.timer.add_checkpoint("Prepared jets")

        # ------------------------------------------------------------#
        # Apply JEC, get JEC and JER variations
        # ------------------------------------------------------------#
        jets = df.Jet

        self.do_jec = False
        if ('data' in dataset) and ('2018' in self.year):
            self.do_jec = True

        if self.do_jec:
            if is_mc:
                factory = self.jet_factory['jec']
            else:
                for run in self.data_runs:
                    if run in dataset:
                        factory = self.jec_factories_data[run]
            jets = factory.build(df.Jet, lazy_cache=df.caches[0])

        if is_mc and self.do_jecunc:
            jets = self.jet_factory['junc'].build(
                jets, lazy_cache=df.caches[0]
            )

        if is_mc and self.do_jerunc:
            jets = self.jet_factory['jer'].build(
                jets, lazy_cache=df.caches[0]
            )

        # TODO: only consider nuisances that are defined in run parameters

        # for some reason jets are doubled, so we need additional filter
        jets = ak.to_pandas(jets).loc[pd.IndexSlice[:, :, 0], :]
        jets.index = jets.index.droplevel('subsubentry')

        new_columns = []
        for v in jets.columns.values:
            if type(v) is tuple:
                new_columns.append(v)
            else:
                new_columns.append(('nominal', '', v))
        jets.columns = new_columns
        jets.columns = pd.MultiIndex.from_tuples(
            jets.columns, names=['Variation', 'Up/Down', 'Variable']
        )

        if self.do_jec:
            jets[('nominal', '', 'pt')] = jets[('nominal', '', 'pt_jec')]
            jets[('nominal', '', 'mass')] = jets[('nominal', '', 'mass_jec')]

        if self.do_jerunc:
            # We use JER corrections only for systematics,
            # not actually applying JER
            jets[('nominal', '', 'pt')] = jets[('nominal', '', 'pt_orig')]
            jets[('nominal', '', 'mass')] = jets[('nominal', '', 'mass_orig')]

        # TODO: JER nuisances
        """
        if is_mc and self.do_jerunc:
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

        """
        if self.timer:
            self.timer.add_checkpoint("Applied JEC/JER (if enabled)")

        # ------------------------------------------------------------#
        # Apply jetID
        # ------------------------------------------------------------#
        jet_id = jets[('nominal', '', 'jetId')]
        jet_qgl = jets[('nominal', '', 'qgl')]

        pass_jet_id = np.ones_like(jet_id, dtype=bool)

        if "loose" in self.parameters["jet_id"]:
            pass_jet_id = (jet_id >= 1)
        elif "tight" in self.parameters["jet_id"]:
            if '2016' in self.year:
                pass_jet_id = (jet_id >= 3)
            else:
                pass_jet_id = (jet_id >= 2)

        jets = jets[pass_jet_id & (jet_qgl > -2)]

        # TODO: clean jets from muons
        """
        mujet = df.Jet.cross(self.muons_all, nested=True)
        _, _, deltar_mujet = delta_r(
            mujet.i0.eta, mujet.i1.eta_raw, mujet.i0.phi,
            mujet.i1.phi_raw)
        deltar_mujet_ok = (
            deltar_mujet > self.parameters["min_dr_mu_jet"]).all()
        jets = jets[deltar_mujet_ok]
        """

        # ------------------------------------------------------------#
        # Calculate other event weights
        # ------------------------------------------------------------#

        if is_mc:
            if self.do_nnlops and ('ggh' in dataset):
                nnlops = NNLOPS_Evaluator('data/NNLOPS_reweight.root')
                nnlopsw = np.ones(numevents, dtype=float)
                if 'amc' in dataset:
                    nnlopsw = nnlops.evaluate(
                        df.HTXS.Higgs_pt, df.HTXS.njets30, "mcatnlo"
                    )
                elif 'powheg' in dataset:
                    nnlopsw = nnlops.evaluate(
                        df.HTXS.Higgs_pt, df.HTXS.njets30, "powheg"
                    )
                weights.add_weight('nnlops', nnlopsw)

            """
            if ('dy' in dataset) and False:  # disable for now
                zpt_weight = np.ones(numevents, dtype=float)
                zpt_weight[two_muons] =\
                    self.evaluator[self.zpt_path](
                        output['dimuon_pt'][two_muons]
                    ).flatten()
                weights.add_weight('zpt_wgt', zpt_weight)
            """

            sf = musf_evaluator(
                self.musf_lookup,
                self.year,
                numevents,
                mu1, mu2
            )

            weights.add_weight_with_variations(
                'muID', sf['muID'], sf['muID_up'], sf['muID_down'])
            weights.add_weight_with_variations(
                'muIso', sf['muIso'], sf['muIso_up'], sf['muIso_down'])
            # weights.add_weight_with_variations(
            #     'muTrig', muTrig['nom'], muTrig['up'], muTrig['down'])

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

            do_thu = (
                ('vbf' in dataset) and
                ('dy' not in dataset) and
                ('nominal' in self.pt_variations)
            )

            if do_thu and ('stage1_1_fine_cat_pTjet30GeV' in df.HTXS.fields):
                for i, name in enumerate(self.sths_names):
                    wgt_up = vbf_uncert_stage_1_1(
                        i,
                        ak.to_numpy(df.HTXS.stage1_1_fine_cat_pTjet30GeV),
                        1.,
                        self.stxs_acc_lookups,
                        self.powheg_xsec_lookup
                    )
                    wgt_down = vbf_uncert_stage_1_1(
                        i,
                        ak.to_numpy(df.HTXS.stage1_1_fine_cat_pTjet30GeV),
                        -1.,
                        self.stxs_acc_lookups,
                        self.powheg_xsec_lookup
                    )
                    weights.add_only_variations(
                        "THU_VBF_"+name, wgt_up, wgt_down
                    )

        if self.timer:
            self.timer.add_checkpoint("Computed event weights")

        # ------------------------------------------------------------#
        # Calculate getJetMass
        # ------------------------------------------------------------#

        output['genJetPairMass'] = 0.0
        if is_mc:
            gjets = df.GenJet
            gleptons = df.GenPart[
                    (abs(df.GenPart.pdgId) == 13) |
                    (abs(df.GenPart.pdgId) == 11) |
                    (abs(df.GenPart.pdgId) == 15)
            ]
            gl_pair = ak.cartesian(
                {'jet': gjets, 'lepton': gleptons},
                axis=1,
                nested=True
            )
            _, _, dr_gl = delta_r(
                    gl_pair['jet'].eta,
                    gl_pair['lepton'].eta,
                    gl_pair['jet'].phi,
                    gl_pair['lepton'].phi
            )
            isolated = ak.all((dr_gl > 0.3), axis=-1)
            gjet1 = ak.to_pandas(gjets[isolated]).loc[
                pd.IndexSlice[:, 0], ['pt', 'eta', 'phi', 'mass']
            ]
            gjet2 = ak.to_pandas(gjets[isolated]).loc[
                pd.IndexSlice[:, 1], ['pt', 'eta', 'phi', 'mass']
            ]
            gjet1.index = gjet1.index.droplevel('subentry')
            gjet2.index = gjet2.index.droplevel('subentry')

            gjsum = p4_sum(gjet1, gjet2)

            output['genJetPairMass'] = gjsum.mass
            output['genJetPairMass'] = output['genJetPairMass'].fillna(0.0)

        # ------------------------------------------------------------#
        # Loop over JEC variations and fill jet variables
        # ------------------------------------------------------------#

        output.columns = pd.MultiIndex.from_product(
            [output.columns, ['']], names=['Variable', 'Variation']
        )
        for v_name in self.pt_variations:
            output_updated = self.jec_loop(
                v_name, is_mc, df, dataset, mask, muons,
                mu1, mu2, jets, weights, numevents, output
            )
            if output_updated is not None:
                output = output_updated
        if self.timer:
            self.timer.add_checkpoint("Completed JEC loop")

        # ------------------------------------------------------------#
        # PDF variations
        # ------------------------------------------------------------#

        if self.do_pdf and is_mc and ('nominal' in self.pt_variations):
            do_pdf = (
                ("dy" in dataset or
                 "ewk" in dataset or
                 "ggh" in dataset or
                 "vbf" in dataset) and
                ('mg' not in dataset)
            )
            if do_pdf:
                pdf_wgts = df.LHEPdfWeight[
                    :, 0:self.parameters["n_pdf_variations"]
                ]
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
                        (1 - 2 * pdf_wgts.std())
                    )

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
                :, pd.IndexSlice['c', 'nominal']
            ].isin(self.channels)
        ]

        output = output[output.r.isin(self.regions)]

        if self.timer:
            self.timer.add_checkpoint("Filled outputs")
            self.timer.summary()
        return output

    def jec_loop(self, variation, is_mc, df, dataset, mask, muons,
                 mu1, mu2, jets, weights, numevents, output):
        weights = copy.deepcopy(weights)

        # What is done here:
        # for each systematic variation of jet pT (including nominal)
        # - selection (with pt cut)
        # - puID depends on pt
        # - picking two highest-pT jets
        # - variables for the jet pair
        # - QGL weights
        # - Soft activity variables
        # - B-tag weighs
        # - Category definitions
        # - Matching jets to genjets

        # ------------------------------------------------------------#
        # Initialize jet-related variables
        # ------------------------------------------------------------#

        variable_names = [
            'jet1_pt', 'jet1_eta', 'jet1_rap', 'jet1_phi', 'jet1_qgl',
            'jet1_jetId', 'jet1_puId',
            'jet2_pt', 'jet2_eta', 'jet2_rap', 'jet2_phi', 'jet2_qgl',
            'jet2_jetId', 'jet2_puId',
            'jj_mass', 'jj_mass_log', 'jj_pt', 'jj_eta', 'jj_phi',
            'jj_dEta', 'jj_dPhi',
            'mmj1_dEta', 'mmj1_dPhi', 'mmj1_dR',
            'mmj2_dEta', 'mmj2_dPhi', 'mmj2_dR',
            'mmj_min_dEta', 'mmj_min_dPhi', 'mmjj_pt',
            'mmjj_eta', 'mmjj_phi', 'mmjj_mass', 'rpt',
            'zeppenfeld', 'll_zstar_log', 'nsoftjets2',
            'nsoftjets5', 'htsoft2', 'htsoft5', 'selection'
        ]

        variables = pd.DataFrame(index=output.index, columns=variable_names)
        variables = variables.fillna(-999.)

        # ------------------------------------------------------------#
        # Select jets for certain pT variation
        # ------------------------------------------------------------#
        if '_up' in variation:
            unc_name = 'JES_'+variation.replace('_up', '')
            jets = jets.loc[:, ([unc_name], ['up'], slice(None))]
        elif '_down' in variation:
            unc_name = variation.replace('_down', '')
            jets = jets.loc[:, ([unc_name], ['down'], slice(None))]
        else:
            jets = jets.loc[:, (['nominal'], slice(None), slice(None))]

        jets.columns = jets.columns.droplevel('Variation')
        jets.columns = jets.columns.droplevel('Up/Down')

        if jets.count().sum() == 0:
            return

        jet_selection = (
            (jets.pt > self.parameters["jet_pt_cut"]) &
            (abs(jets.eta) < self.parameters["jet_eta_cut"])
        )

        # ------------------------------------------------------------#
        # Calculate PUID scale factors and apply PUID
        # ------------------------------------------------------------#

        # Jet PU ID
        jet_puid_opt = self.parameters["jet_puid"]
        puId = jets.puId17 if self.year == "2017" else jets.puId
        jet_puid_wps = {
            "loose": (puId >= 4) | (jets.pt > 50),
            "medium": (puId >= 6) | (jets.pt > 50),
            "tight": (puId >= 7) | (jets.pt > 50),
        }
        jet_puid = np.ones_like(jets.pt.values)
        if jet_puid_opt in ["loose", "medium", "tight"]:
            jet_puid = jet_puid_wps[jet_puid_opt]
        elif "2017corrected" in jet_puid_opt:
            eta_window = (
                (abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0)
            )
            jet_puid = (
                (eta_window & (puId >= 7)) |
                ((~eta_window) & jet_puid_wps['loose'])
            )

        # Jet PUID scale factors
        # if is_mc and False:  # disable for now
        #     puid_weight = puid_weights(
        #         self.evaluator, self.year, jets, pt_name,
        #         jet_puid_opt, jet_puid, numevents
        #     )
        #     weights.add_weight('puid_wgt', puid_weight)

        jets['selection'] = jet_selection & jet_puid

        if self.timer:
            self.timer.add_checkpoint("Selected jets")

        # ------------------------------------------------------------#
        # Fill jet-related variables
        # ------------------------------------------------------------#

        njets = jets[jets.selection].reset_index()\
            .groupby('entry')['subentry'].nunique()
        variables['njets'] = njets

        one_jet = (jets.selection & (njets > 0))
        two_jets = (jets.selection & (njets > 1))

        # Sort jets by pT and reset their numbering in an event
        jets = jets.sort_values(
            ['entry', 'pt'], ascending=[True, False]
        )
        jets.index = pd.MultiIndex.from_arrays(
            [
                jets.index.get_level_values(0),
                jets.groupby(level=0).cumcount()
            ],
            names=['entry', 'subentry']
        )

        # Select two jets with highest pT
        jet1 = jets.loc[pd.IndexSlice[:, 0], :]
        jet2 = jets.loc[pd.IndexSlice[:, 1], :]
        jet1.index = jet1.index.droplevel('subentry')
        jet2.index = jet2.index.droplevel('subentry')

        # Fill single jet variables
        for v in ['pt', 'eta', 'phi', 'qgl', 'jetId', 'puId']:
            variables[f'jet1_{v}'] = jet1[v]
            variables[f'jet2_{v}'] = jet2[v]

        variables.jet1_rap = rapidity(jet1)
        variables.jet2_rap = rapidity(jet2)

        # Fill dijet variables
        jj = p4_sum(jet1, jet2)
        for v in ['pt', 'eta', 'phi', 'mass']:
            variables[f'jj_{v}'] = jj[v]

        variables.jj_mass_log = np.log(variables.jj_mass)

        variables.jj_dEta, variables.jj_dPhi, _ = delta_r(
            variables.jet1_eta,
            variables.jet2_eta,
            variables.jet1_phi,
            variables.jet2_phi
        )

        # Fill dimuon-dijet system variables
        mm_columns = [
            'dimuon_pt', 'dimuon_eta', 'dimuon_phi', 'dimuon_mass'
        ]
        jj_columns = [
            'jj_pt', 'jj_eta', 'jj_phi', 'jj_mass'
        ]

        dimuons = output.loc[:, mm_columns]
        dijets = variables.loc[:, jj_columns]

        # careful with renaming
        dimuons.columns = ['mass', 'pt', 'eta', 'phi']
        dijets.columns = ['pt', 'eta', 'phi', 'mass']

        mmjj = p4_sum(dimuons, dijets)
        for v in ['pt', 'eta', 'phi', 'mass']:
            variables[f'mmjj_{v}'] = mmjj[v]

        variables.zeppenfeld = (
            output.dimuon_eta - 0.5 * (
                variables.jet1_eta +
                variables.jet2_eta
            )
        )

        variables.rpt = variables.mmjj_pt / (
            output.dimuon_pt +
            variables.jet1_pt +
            variables.jet2_pt
        )

        ll_ystar = (
            output.dimuon_rap -
            (variables.jet1_rap + variables.jet2_rap) / 2
        )

        ll_zstar = abs(
            ll_ystar / (
                variables.jet1_rap - variables.jet2_rap
            )
        )

        variables.ll_zstar_log = np.log(ll_zstar)

        variables.mmj1_dEta,\
            variables.mmj1_dPhi,\
            variables.mmj1_dR = delta_r(
                output.dimuon_eta,
                variables.jet1_eta,
                output.dimuon_phi,
                variables.jet1_phi
            )

        variables.mmj2_dEta,\
            variables.mmj2_dPhi,\
            variables.mmj2_dR = delta_r(
                output.dimuon_eta,
                variables.jet2_eta,
                output.dimuon_phi,
                variables.jet2_phi
            )

        variables.mmj_min_dEta = np.where(
            variables.mmj1_dEta,
            variables.mmj2_dEta,
            (variables.mmj1_dEta < variables.mmj2_dEta)
        )

        variables.mmj_min_dPhi = np.where(
            variables.mmj1_dPhi,
            variables.mmj2_dPhi,
            (variables.mmj1_dPhi < variables.mmj2_dPhi)
        )

        if self.timer:
            self.timer.add_checkpoint("Filled jet variables")

        # ------------------------------------------------------------#
        # Fill soft activity jet variables
        # ------------------------------------------------------------#

        # variables.nsoftjets2,\
        #     variables.htsoft2 = self.get_softjet_vars(
        #         df, df.SoftActivityJet, 2, muons, mu1, mu2,
        #         jets, jet1, jet2, two_muons, one_jet, two_jets
        #     )

        # variables.nsoftjets5,\
        #     variables.htsoft5 = self.get_softjet_vars(
        #         df, df.SoftActivityJet, 5, muons, mu1, mu2,
        #         jets, jet1, jet2, two_muons, one_jet, two_jets
        #     )

        # if self.timer:
        #     self.timer.add_checkpoint("Calculated SA variables")

        # ------------------------------------------------------------#
        # Apply remaining cuts
        # ------------------------------------------------------------#

        # Cut has to be defined here because we will use it in
        # b-tag weights calculation
        vbf_cut = (
            (variables.jj_mass > 400) &
            (variables.jj_dEta > 2.5) &
            (jet1.pt > 35)
        )

        # ------------------------------------------------------------#
        # Calculate QGL weights, btag SF and apply btag veto
        # ------------------------------------------------------------#

        if is_mc:
            isHerwig = ('herwig' in dataset)

            qgl = pd.DataFrame(
                index=output.index, columns=['wgt', 'wgt_down']
            ).fillna(1.0)

            qgl1 = qgl_weights(jet1, isHerwig).fillna(1.0)
            qgl2 = qgl_weights(jet2, isHerwig).fillna(1.0)
            qgl.wgt *= qgl1 * qgl2

            qgl.wgt[variables.njets == 1] = 1.
            selected = output.event_selection & (njets > 2)
            qgl.wgt = qgl.wgt / qgl.wgt[selected].mean()

            weights.add_weight_with_variations(
                'qgl_wgt', qgl.wgt,
                up=qgl.wgt*qgl.wgt, down=qgl.wgt_down
            )

        # TODO: fix
        """
        bjet_sel_mask = output.event_selection & two_jets & vbf_cut
        # Btag weight
        btag_wgt = np.ones(numevents)
        if is_mc:
            systs = self.btag_systs if 'nominal' in variation else []
            btag_wgt, btag_syst = btag_weights(
                self, self.btag_lookup, systs, jets,
                weights, bjet_sel_mask, numevents
            )

            weights.add_weight('btag_wgt', btag_wgt)
            for name, bs in btag_syst.items():
                up = bs[0]
                down = bs[1]
                weights.add_only_variations(
                    f'btag_wgt_{name}', up, down
                )
        """

        # Separate from ttH and VH phase space
        variables['nBtagLoose'] = jets[
            jets.selection &
            (jets.btagDeepB > self.parameters["btag_loose_wp"]) &
            (abs(jets.eta) < 2.5)
        ].reset_index().groupby('entry')['subentry'].nunique()

        variables['nBtagMedium'] = jets[
            jets.selection &
            (jets.btagDeepB > self.parameters["btag_medium_wp"]) &
            (abs(jets.eta) < 2.5)
        ].reset_index().groupby('entry')['subentry'].nunique()
        variables.nBtagLoose = variables.nBtagLoose.fillna(0.0)
        variables.nBtagMedium = variables.nBtagMedium.fillna(0.0)

        variables.selection = (
            output.event_selection &
            (variables.nBtagLoose < 2) &
            (variables.nBtagMedium < 1)
        )

        if self.timer:
            self.timer.add_checkpoint(
                "Applied b-jet SF and b-tag veto"
            )

        # ------------------------------------------------------------#
        # Define categories
        # ------------------------------------------------------------#
        variables['c'] = ''
        variables.c[
            variables.selection & (variables.njets < 2)] = 'ggh_01j'
        variables.c[
            variables.selection &
            (variables.njets >= 2) & (~vbf_cut)] = 'ggh_2j'
        variables.c[
            variables.selection &
            (variables.njets >= 2) & vbf_cut] = 'vbf'

        if 'dy' in dataset:
            two_jets_matched = np.zeros(numevents, dtype=bool)
            matched1 =\
                (jet1.matched_genjet.counts > 0)[two_jets[one_jet]]
            matched2 = (jet2.matched_genjet.counts > 0)
            two_jets_matched[two_jets] = matched1 & matched2
            variables.c[
                variables.selection &
                (variables.njets >= 2) &
                vbf_cut & (~two_jets_matched)] = 'vbf_01j'
            variables.c[
                variables.selection &
                (variables.njets >= 2) &
                vbf_cut & two_jets_matched] = 'vbf_2j'

        # --------------------------------------------------------------#
        # Fill outputs
        # --------------------------------------------------------------#

        variables.update({'wgt_nominal': weights.get_weight('nominal')})

        # All variables are affected by jet pT because of jet selections:
        # a jet may or may not be selected depending on pT variation,
        # so even variables that don't directly depend on jet pT
        # (e.g. QGL) must be saved for all pT variations.

        for key, val in variables.items():
            output.loc[:, pd.IndexSlice[key, variation]] = val

        return output

    def get_softjet_vars(self, df, softjets, cutoff, muons, mu1, mu2,
                         jets, jet1, jet2, two_muons, one_jet, two_jets):
        saj_df = ak.to_pandas(df.SoftActivityJet)
        saj_df['mass'] = 0.0
        print(saj_df)
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

        sj_j = softjets.cross(jets, nested=True)
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
