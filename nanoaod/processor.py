import awkward
import awkward as ak
import numpy as np

# np.set_printoptions(threshold=sys.maxsize)
import pandas as pd

import coffea.processor as processor
from coffea.lookup_tools import extractor
from coffea.lookup_tools import txt_converters, rochester_lookup
from coffea.btag_tools import BTagScaleFactor
from coffea.lumi_tools import LumiMask

from python.timer import Timer
from nanoaod.weights import Weights
from nanoaod.corrections.pu_reweight import pu_lookups, pu_evaluator
from nanoaod.corrections.l1prefiring_weights import l1pf_weights
from nanoaod.corrections.rochester import apply_roccor
from nanoaod.corrections.fsr_recovery import fsr_recovery
from nanoaod.corrections.geofit import apply_geofit
from nanoaod.corrections.jec import jec_factories, apply_jec
from nanoaod.corrections.lepton_sf import musf_lookup, musf_evaluator
from nanoaod.corrections.nnlops import nnlops_weights
from nanoaod.corrections.stxs_uncert import stxs_uncert, stxs_lookups
from nanoaod.corrections.lhe_weights import lhe_weights
from nanoaod.corrections.qgl_weights import qgl_weights
from nanoaod.corrections.btag_weights import btag_weights

# from nanoaod.corrections.puid_weights import puid_weights

from nanoaod.muons import fill_muons
from nanoaod.jets import prepare_jets, fill_jets, fill_softjets
from nanoaod.jets import jet_id, jet_puid, gen_jet_pair_mass

from config.parameters import parameters
from config.variables import variables


class DimuonProcessor(processor.ProcessorABC):
    def __init__(self, **kwargs):
        self.samp_info = kwargs.pop("samp_info", None)
        do_timer = kwargs.pop("do_timer", False)
        self.pt_variations = kwargs.pop("pt_variations", ["nominal"])
        self.do_btag_syst = kwargs.pop("do_btag_syst", True)
        self.apply_to_output = kwargs.pop("apply_to_output", None)

        if self.samp_info is None:
            print("Samples info missing!")
            return

        self._accumulator = processor.defaultdict_accumulator(int)

        self.year = self.samp_info.year
        self.parameters = {k: v[self.year] for k, v in parameters.items()}

        self.do_roccor = True
        self.do_fsr = True
        self.do_geofit = True
        self.auto_pu = True
        self.do_nnlops = True
        self.do_pdf = True

        self.timer = Timer("global") if do_timer else None

        self._columns = self.parameters["proc_columns"]

        # --- Define regions and channels used in the analysis ---#
        self.regions = ["z-peak", "h-sidebands", "h-peak"]
        # self.channels = ['ggh_01j', 'ggh_2j', 'vbf']
        self.channels = ["vbf", "vbf_01j", "vbf_2j"]

        self.lumi_weights = self.samp_info.lumi_weights

        self.sths_names = [
            "Yield",
            "PTH200",
            "Mjj60",
            "Mjj120",
            "Mjj350",
            "Mjj700",
            "Mjj1000",
            "Mjj1500",
            "PTH25",
            "JET01",
        ]

        if self.do_btag_syst:
            self.btag_systs = [
                "jes",
                "lf",
                "hfstats1",
                "hfstats2",
                "cferr1",
                "cferr2",
                "hf",
                "lfstats1",
                "lfstats2",
            ]
        else:
            self.btag_systs = []

        self.vars_to_save = set([v.name for v in variables])

        # Prepare lookup tables for all kinds of corrections
        self.prepare_lookups()

        # Look at variation names and see if we need to enable
        # calculation of JEC or JER uncertainties
        self.do_jecunc = False
        self.do_jerunc = False
        for ptvar in self.pt_variations:
            ptvar_ = ptvar.replace("_up", "").replace("_down", "")
            if ptvar_ in self.parameters["jec_unc_to_consider"]:
                self.do_jecunc = True
            jers = ["jer1", "jer2", "jer3", "jer4", "jer5", "jer6"]
            if ptvar_ in jers:
                self.do_jerunc = True

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    def process(self, df):
        # Initialize timer
        if self.timer:
            self.timer.update()

        # Dataset name (see definitions in config/datasets.py)
        dataset = df.metadata["dataset"]
        is_mc = "data" not in dataset
        numevents = len(df)

        # ------------------------------------------------------------#
        # Apply HLT, lumimask, genweights, PU weights
        # and L1 prefiring weights
        # ------------------------------------------------------------#

        # All variables that we want to save
        # will be collected into the 'output' dataframe
        output = pd.DataFrame({"run": df.run, "event": df.event})
        output.index.name = "entry"
        output["npv"] = df.PV.npvs
        output["met"] = df.MET.pt

        # Separate dataframe to keep track on weights
        # and their systematic variations
        weights = Weights(output)

        if is_mc:
            # For MC: Apply gen.weights, pileup weights, lumi weights,
            # L1 prefiring weights
            mask = np.ones(numevents, dtype=bool)
            genweight = df.genWeight
            weights.add_weight("genwgt", genweight)
            weights.add_weight("lumi", self.lumi_weights[dataset])

            pu_wgts = pu_evaluator(
                self.pu_lookups,
                self.parameters,
                numevents,
                np.array(df.Pileup.nTrueInt),
                self.auto_pu,
            )
            weights.add_weight("pu_wgt", pu_wgts, how="all")

            if self.parameters["do_l1prefiring_wgts"]:
                if "L1PreFiringWeight" in df.fields:
                    l1pfw = l1pf_weights(df)
                    weights.add_weight("l1prefiring_wgt", l1pfw, how="all")
                else:
                    weights.add_weight("l1prefiring_wgt", how="dummy_vars")

        else:
            # For Data: apply Lumi mask
            lumi_info = LumiMask(self.parameters["lumimask"])
            mask = lumi_info(df.run, df.luminosityBlock)

        # Apply HLT to both Data and MC
        hlt_columns = [c for c in self.parameters["hlt"] if c in df.HLT.fields]
        hlt = ak.to_pandas(df.HLT[hlt_columns])
        if len(hlt_columns) == 0:
            hlt = False
        else:
            hlt = hlt[hlt_columns].sum(axis=1)

        if self.timer:
            self.timer.add_checkpoint("HLT, lumimask, PU weights")

        # ------------------------------------------------------------#
        # Update muon kinematics with Rochester correction,
        # FSR recovery and GeoFit correction
        # Raw pT and eta are stored to be used in event selection
        # ------------------------------------------------------------#

        # Save raw variables before computing any corrections
        df["Muon", "pt_raw"] = df.Muon.pt
        df["Muon", "eta_raw"] = df.Muon.eta
        df["Muon", "phi_raw"] = df.Muon.phi
        df["Muon", "pfRelIso04_all_raw"] = df.Muon.pfRelIso04_all

        # Rochester correction
        if self.do_roccor:
            apply_roccor(df, self.roccor_lookup, is_mc)
            df["Muon", "pt"] = df.Muon.pt_roch

            # variations will be in branches pt_roch_up and pt_roch_down
            # muons_pts = {
            #     'nominal': df.Muon.pt,
            #     'roch_up':df.Muon.pt_roch_up,
            #     'roch_down':df.Muon.pt_roch_down
            # }

        # for ...
        if True:  # indent reserved for loop over muon pT variations
            # According to HIG-19-006, these variations have negligible
            # effect on significance, but it's better to have them
            # implemented in the future

            # FSR recovery
            if self.do_fsr:
                has_fsr = fsr_recovery(df)
                df["Muon", "pt"] = df.Muon.pt_fsr
                df["Muon", "eta"] = df.Muon.eta_fsr
                df["Muon", "phi"] = df.Muon.phi_fsr
                df["Muon", "pfRelIso04_all"] = df.Muon.iso_fsr

            # if FSR was applied, 'pt_fsr' will be corrected pt
            # if FSR wasn't applied, just copy 'pt' to 'pt_fsr'
            df["Muon", "pt_fsr"] = df.Muon.pt

            # GeoFit correction
            if self.do_geofit and ("dxybs" in df.Muon.fields):
                apply_geofit(df, self.year, ~has_fsr)
                df["Muon", "pt"] = df.Muon.pt_fsr

            if self.timer:
                self.timer.add_checkpoint("Muon corrections")

            # --- conversion from awkward to pandas --- #
            muon_columns = [
                "pt",
                "pt_fsr",
                "eta",
                "phi",
                "charge",
                "ptErr",
                "mass",
                "pt_raw",
                "eta_raw",
                "pfRelIso04_all",
            ] + [self.parameters["muon_id"]]
            muons = ak.to_pandas(df.Muon[muon_columns])

            # --------------------------------------------------------#
            # Select muons that pass pT, eta, isolation cuts,
            # muon ID and quality flags
            # Select events with 2 OS muons, no electrons,
            # passing quality cuts and at least one good PV
            # --------------------------------------------------------#

            # Apply event quality flags
            flags = ak.to_pandas(df.Flag[self.parameters["event_flags"]])
            flags = flags[self.parameters["event_flags"]].product(axis=1)
            muons["pass_flags"] = True
            if self.parameters["muon_flags"]:
                muons["pass_flags"] = muons[self.parameters["muon_flags"]].product(
                    axis=1
                )

            # Define baseline muon selection (applied to pandas DF!)
            muons["selection"] = (
                (muons.pt_raw > self.parameters["muon_pt_cut"])
                & (abs(muons.eta_raw) < self.parameters["muon_eta_cut"])
                & (muons.pfRelIso04_all < self.parameters["muon_iso_cut"])
                & muons[self.parameters["muon_id"]]
                & muons.pass_flags
            )

            # Count muons
            nmuons = (
                muons[muons.selection]
                .reset_index()
                .groupby("entry")["subentry"]
                .nunique()
            )

            # Find opposite-sign muons
            mm_charge = muons.loc[muons.selection, "charge"].groupby("entry").prod()

            # Veto events with good quality electrons
            electrons = df.Electron[
                (df.Electron.pt > self.parameters["electron_pt_cut"])
                & (abs(df.Electron.eta) < self.parameters["electron_eta_cut"])
                & (df.Electron[self.parameters["electron_id"]] == 1)
            ]
            electron_veto = ak.to_numpy(ak.count(electrons.pt, axis=1) == 0)

            # Find events with at least one good primary vertex
            good_pv = ak.to_pandas(df.PV).npvsGood > 0

            # Define baseline event selection
            output["two_muons"] = nmuons == 2
            output["event_selection"] = (
                mask
                & (hlt > 0)
                & (flags > 0)
                & (nmuons == 2)
                & (mm_charge == -1)
                & electron_veto
                & good_pv
            )

            # --------------------------------------------------------#
            # Select two leading-pT muons
            # --------------------------------------------------------#

            # Find pT-leading and subleading muons
            # This is slow for large chunk size.
            # Consider reimplementing using sort_values().groupby().nth()
            # or sort_values().drop_duplicates()
            # or using Numba
            # https://stackoverflow.com/questions/50381064/select-the-max-row-per-group-pandas-performance-issue
            muons = muons[muons.selection & (nmuons == 2)]
            mu1 = muons.loc[muons.pt.groupby("entry").idxmax()]
            mu2 = muons.loc[muons.pt.groupby("entry").idxmin()]
            mu1.index = mu1.index.droplevel("subentry")
            mu2.index = mu2.index.droplevel("subentry")

            # --------------------------------------------------------#
            # Select events with muons passing leading pT cut
            # and trigger matching (trig match not done in final vrsn)
            # --------------------------------------------------------#

            # Events where there is at least one muon passing
            # leading muon pT cut
            pass_leading_pt = mu1.pt_raw > self.parameters["muon_leading_pt"]

            # update event selection with leading muon pT cut
            output["pass_leading_pt"] = pass_leading_pt
            output["event_selection"] = output.event_selection & output.pass_leading_pt

            # --------------------------------------------------------#
            # Fill dimuon and muon variables
            # --------------------------------------------------------#

            fill_muons(self, output, mu1, mu2, is_mc)

            if self.timer:
                self.timer.add_checkpoint("Event & muon selection")

        # ------------------------------------------------------------#
        # Prepare jets
        # ------------------------------------------------------------#

        prepare_jets(df, is_mc)

        # ------------------------------------------------------------#
        # Apply JEC, get JEC and JER variations
        # ------------------------------------------------------------#

        jets = df.Jet

        self.do_jec = False

        # We only need to reapply JEC for 2018 data
        # (unless new versions of JEC are released)
        if ("data" in dataset) and ("2018" in self.year):
            self.do_jec = True

        jets = apply_jec(
            df,
            jets,
            dataset,
            is_mc,
            self.year,
            self.do_jec,
            self.do_jecunc,
            self.do_jerunc,
            self.jec_factories,
            self.jec_factories_data,
        )

        # ------------------------------------------------------------#
        # Calculate other event weights
        # ------------------------------------------------------------#

        if is_mc:
            do_nnlops = self.do_nnlops and ("ggh" in dataset)
            if do_nnlops:
                nnlopsw = nnlops_weights(df, numevents, self.parameters, dataset)
                weights.add_weight("nnlops", nnlopsw)
            else:
                weights.add_weight("nnlops", how="dummy")
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            # do_zpt = ('dy' in dataset)
            #
            # if do_zpt:
            #     zpt_weight = np.ones(numevents, dtype=float)
            #     zpt_weight[two_muons] =\
            #         self.evaluator[self.zpt_path](
            #             output['dimuon_pt'][two_muons]
            #         ).flatten()
            #     weights.add_weight('zpt_wgt', zpt_weight)
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            do_musf = True
            if do_musf:
                muID, muIso, muTrig = musf_evaluator(
                    self.musf_lookup, self.year, numevents, mu1, mu2
                )
                weights.add_weight("muID", muID, how="all")
                weights.add_weight("muIso", muIso, how="all")
                weights.add_weight("muTrig", muTrig, how="all")
            else:
                weights.add_weight("muID", how="dummy_all")
                weights.add_weight("muIso", how="dummy_all")
                weights.add_weight("muTrig", how="dummy_all")
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            do_lhe = (
                ("LHEScaleWeight" in df.fields)
                and ("LHEPdfWeight" in df.fields)
                and ("nominal" in self.pt_variations)
            )
            if do_lhe:
                lhe_ren, lhe_fac = lhe_weights(df, output, dataset, self.year)
                weights.add_weight("LHERen", lhe_ren, how="only_vars")
                weights.add_weight("LHEFac", lhe_fac, how="only_vars")
            else:
                weights.add_weight("LHERen", how="dummy_vars")
                weights.add_weight("LHEFac", how="dummy_vars")
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            do_thu = (
                ("vbf" in dataset)
                and ("dy" not in dataset)
                and ("nominal" in self.pt_variations)
                and ("stage1_1_fine_cat_pTjet30GeV" in df.HTXS.fields)
            )
            if do_thu:
                for i, name in enumerate(self.sths_names):
                    wgt_up = stxs_uncert(
                        i,
                        ak.to_numpy(df.HTXS.stage1_1_fine_cat_pTjet30GeV),
                        1.0,
                        self.stxs_acc_lookups,
                        self.powheg_xsec_lookup,
                    )
                    wgt_down = stxs_uncert(
                        i,
                        ak.to_numpy(df.HTXS.stage1_1_fine_cat_pTjet30GeV),
                        -1.0,
                        self.stxs_acc_lookups,
                        self.powheg_xsec_lookup,
                    )
                    thu_wgts = {"up": wgt_up, "down": wgt_down}
                    weights.add_weight("THU_VBF_" + name, thu_wgts, how="only_vars")
            else:
                for i, name in enumerate(self.sths_names):
                    weights.add_weight("THU_VBF_" + name, how="dummy_vars")
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
            do_pdf = (
                self.do_pdf
                and ("nominal" in self.pt_variations)
                and (
                    "dy" in dataset
                    or "ewk" in dataset
                    or "ggh" in dataset
                    or "vbf" in dataset
                )
                and ("mg" not in dataset)
            )
            if "2016" in self.year:
                max_replicas = 0
                if "dy" in dataset:
                    max_replicas = 100
                elif "ewk" in dataset:
                    max_replicas = 33
                else:
                    max_replicas = 100
                if do_pdf:
                    pdf_wgts = df.LHEPdfWeight[
                        :, 0 : self.parameters["n_pdf_variations"]
                    ]
                for i in range(100):
                    if (i < max_replicas) and do_pdf:
                        output[f"pdf_mcreplica{i}"] = pdf_wgts[:, i]
                    else:
                        output[f"pdf_mcreplica{i}"] = np.nan
            else:
                if do_pdf:
                    pdf_wgts = df.LHEPdfWeight[
                        :, 0 : self.parameters["n_pdf_variations"]
                    ][0]
                    pdf_wgts = np.array(pdf_wgts)
                    pdf_vars = {
                        "up": (1 + 2 * pdf_wgts.std()),
                        "down": (1 - 2 * pdf_wgts.std()),
                    }
                    weights.add_weight("pdf_2rms", pdf_vars, how="only_vars")
                else:
                    weights.add_weight("pdf_2rms", how="dummy_vars")
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #

        # ------------------------------------------------------------#
        # Calculate getJetMass and split DY
        # ------------------------------------------------------------#

        output["genJetPairMass"] = 0.0
        is_dy = ("dy_m105_160_vbf_amc" in dataset) or ("dy_m105_160_amc" in dataset)
        if is_mc and is_dy:
            gj_pair_mass = gen_jet_pair_mass(df)
            if gj_pair_mass is not None:
                output["genJetPairMass"] = gj_pair_mass
                output["genJetPairMass"] = output["genJetPairMass"].fillna(0.0)
                if "dy_m105_160_vbf_amc" in dataset:
                    output.event_selection = output.event_selection & (
                        output.genJetPairMass > 350
                    )
                if "dy_m105_160_amc" in dataset:
                    output.event_selection = output.event_selection & (
                        output.genJetPairMass <= 350
                    )

        # ------------------------------------------------------------#
        # Loop over JEC variations and fill jet variables
        # ------------------------------------------------------------#

        output.columns = pd.MultiIndex.from_product(
            [output.columns, [""]], names=["Variable", "Variation"]
        )

        if self.timer:
            self.timer.add_checkpoint("Jet preparation & event weights")

        for v_name in self.pt_variations:
            output_updated = self.jet_loop(
                v_name,
                is_mc,
                df,
                dataset,
                mask,
                muons,
                mu1,
                mu2,
                jets,
                weights,
                numevents,
                output,
            )
            if output_updated is not None:
                output = output_updated

        if self.timer:
            self.timer.add_checkpoint("Jet loop")

        # ------------------------------------------------------------#
        # Fill outputs
        # ------------------------------------------------------------#
        mass = output.dimuon_mass
        output["r"] = None
        output.loc[((mass > 76) & (mass < 106)), "r"] = "z-peak"
        output.loc[
            ((mass > 110) & (mass < 115.03)) | ((mass > 135.03) & (mass < 150)), "r"
        ] = "h-sidebands"
        output.loc[((mass > 115.03) & (mass < 135.03)), "r"] = "h-peak"
        output["s"] = dataset
        output["year"] = int(self.year)

        for wgt in weights.df.columns:
            skip_saving = (
                ("nominal" not in wgt) and ("up" not in wgt) and ("down" not in wgt)
            )
            if skip_saving:
                continue
            output[f"wgt_{wgt}"] = weights.get_weight(wgt)

        columns_to_save = [
            c
            for c in output.columns
            if (c[0] in self.vars_to_save)
            or ("wgt_" in c[0])
            or ("mcreplica" in c[0])
            or (c[0] in ["c", "r", "s", "year"])
        ]
        output = output.loc[output.event_selection, columns_to_save]
        output = output.reindex(sorted(output.columns), axis=1)

        try:
            output = output[
                output.loc[:, pd.IndexSlice["c", "nominal"]].isin(self.channels)
            ]
        except Exception:
            pass

        output.columns = [" ".join(col).strip() for col in output.columns.values]

        output = output[output.r.isin(self.regions)]

        to_return = None
        if self.apply_to_output is None:
            to_return = output
        else:
            self.apply_to_output(output)
            to_return = self.accumulator.identity()

        if self.timer:
            self.timer.add_checkpoint("Saving outputs")
            self.timer.summary()

        return to_return

    def jet_loop(
        self,
        variation,
        is_mc,
        df,
        dataset,
        mask,
        muons,
        mu1,
        mu2,
        jets,
        weights,
        numevents,
        output,
    ):
        # weights = copy.deepcopy(weights)

        if not is_mc and variation != "nominal":
            return

        variables = pd.DataFrame(index=output.index)

        jet_columns = [
            "pt",
            "eta",
            "phi",
            "jetId",
            "qgl",
            "puId",
            "mass",
            "btagDeepB",
        ]
        if "puId17" in df.Jet.fields:
            jet_columns += ["puId17"]
        if is_mc:
            jet_columns += ["partonFlavour", "hadronFlavour"]
        if variation == "nominal":
            if self.do_jec:
                jet_columns += ["pt_jec", "mass_jec"]
            if is_mc and self.do_jerunc:
                jet_columns += ["pt_orig", "mass_orig"]

        # Find jets that have selected muons within dR<0.4 from them
        matched_mu_pt = jets.matched_muons.pt_fsr
        matched_mu_iso = jets.matched_muons.pfRelIso04_all
        matched_mu_id = jets.matched_muons[self.parameters["muon_id"]]
        matched_mu_pass = (
            (matched_mu_pt > self.parameters["muon_pt_cut"])
            & (matched_mu_iso < self.parameters["muon_iso_cut"])
            & matched_mu_id
        )
        clean = ~(
            ak.to_pandas(matched_mu_pass)
            .astype(float)
            .fillna(0.0)
            .groupby(level=[0, 1])
            .sum()
            .astype(bool)
        )

        # if self.timer:
        #     self.timer.add_checkpoint("Clean jets from matched muons")

        # Select particular JEC variation
        if "_up" in variation:
            unc_name = "JES_" + variation.replace("_up", "")
            if unc_name not in jets.fields:
                return
            jets = jets[unc_name]["up"][jet_columns]
        elif "_down" in variation:
            unc_name = "JES_" + variation.replace("_down", "")
            if unc_name not in jets.fields:
                return
            jets = jets[unc_name]["down"][jet_columns]
        else:
            jets = jets[jet_columns]

        # --- conversion from awkward to pandas --- #
        jets = ak.to_pandas(jets)

        if jets.index.nlevels == 3:
            # sometimes there are duplicates?
            jets = jets.loc[pd.IndexSlice[:, :, 0], :]
            jets.index = jets.index.droplevel("subsubentry")

        if variation == "nominal":
            # Update pt and mass if JEC was applied
            if self.do_jec:
                jets["pt"] = jets["pt_jec"]
                jets["mass"] = jets["mass_jec"]

            # We use JER corrections only for systematics, so we shouldn't
            # update the kinematics. Use original values,
            # unless JEC were applied.
            if is_mc and self.do_jerunc and not self.do_jec:
                jets["pt"] = jets["pt_orig"]
                jets["mass"] = jets["mass_orig"]

        # ------------------------------------------------------------#
        # Apply jetID and PUID
        # ------------------------------------------------------------#

        pass_jet_id = jet_id(jets, self.parameters, self.year)
        pass_jet_puid = jet_puid(jets, self.parameters, self.year)

        # Jet PUID scale factors
        # if is_mc and False:  # disable for now
        #     puid_weight = puid_weights(
        #         self.evaluator, self.year, jets, pt_name,
        #         jet_puid_opt, jet_puid, numevents
        #     )
        #     weights.add_weight('puid_wgt', puid_weight)

        # ------------------------------------------------------------#
        # Select jets
        # ------------------------------------------------------------#
        jets["clean"] = clean

        jet_selection = (
            pass_jet_id
            & pass_jet_puid
            & (jets.qgl > -2)
            & jets.clean
            & (jets.pt > self.parameters["jet_pt_cut"])
            & (abs(jets.eta) < self.parameters["jet_eta_cut"])
        )

        jets = jets[jet_selection]

        # if self.timer:
        #     self.timer.add_checkpoint("Selected jets")

        # ------------------------------------------------------------#
        # Fill jet-related variables
        # ------------------------------------------------------------#

        njets = jets.reset_index().groupby("entry")["subentry"].nunique()
        variables["njets"] = njets

        # one_jet = (njets > 0)
        two_jets = njets > 1

        # Sort jets by pT and reset their numbering in an event
        jets = jets.sort_values(["entry", "pt"], ascending=[True, False])
        jets.index = pd.MultiIndex.from_arrays(
            [jets.index.get_level_values(0), jets.groupby(level=0).cumcount()],
            names=["entry", "subentry"],
        )

        # Select two jets with highest pT
        try:
            jet1 = jets.loc[pd.IndexSlice[:, 0], :]
            jet2 = jets.loc[pd.IndexSlice[:, 1], :]
            jet1.index = jet1.index.droplevel("subentry")
            jet2.index = jet2.index.droplevel("subentry")
        except Exception:
            return

        fill_jets(output, variables, jet1, jet2)

        # if self.timer:
        #     self.timer.add_checkpoint("Filled jet variables")

        # ------------------------------------------------------------#
        # Fill soft activity jet variables
        # ------------------------------------------------------------#

        # Effect of changes in jet acceptance should be negligible,
        # no need to calcluate this for each jet pT variation
        if variation == "nominal":
            fill_softjets(df, output, variables, 2)
            fill_softjets(df, output, variables, 5)

            # if self.timer:
            #     self.timer.add_checkpoint("Calculated SA variables")

        # ------------------------------------------------------------#
        # Apply remaining cuts
        # ------------------------------------------------------------#

        # Cut has to be defined here because we will use it in
        # b-tag weights calculation
        vbf_cut = (variables.jj_mass > 400) & (variables.jj_dEta > 2.5) & (jet1.pt > 35)

        # ------------------------------------------------------------#
        # Calculate QGL weights, btag SF and apply btag veto
        # ------------------------------------------------------------#

        if is_mc and variation == "nominal":
            # --- QGL weights --- #
            isHerwig = "herwig" in dataset

            qgl_wgts = qgl_weights(jet1, jet2, isHerwig, output, variables, njets)
            weights.add_weight("qgl_wgt", qgl_wgts, how="all")

            # --- Btag weights --- #
            bjet_sel_mask = output.event_selection & two_jets & vbf_cut

            btag_wgt, btag_syst = btag_weights(
                self, self.btag_lookup, self.btag_systs, jets, weights, bjet_sel_mask
            )
            weights.add_weight("btag_wgt", btag_wgt)

            # --- Btag weights variations --- #
            for name, bs in btag_syst.items():
                weights.add_weight(f"btag_wgt_{name}", bs, how="only_vars")

            # if self.timer:
            #     self.timer.add_checkpoint(
            #         "Applied QGL and B-tag weights"
            #     )

        # Separate from ttH and VH phase space
        variables["nBtagLoose"] = (
            jets[
                (jets.btagDeepB > self.parameters["btag_loose_wp"])
                & (abs(jets.eta) < 2.5)
            ]
            .reset_index()
            .groupby("entry")["subentry"]
            .nunique()
        )

        variables["nBtagMedium"] = (
            jets[
                (jets.btagDeepB > self.parameters["btag_medium_wp"])
                & (abs(jets.eta) < 2.5)
            ]
            .reset_index()
            .groupby("entry")["subentry"]
            .nunique()
        )
        variables.nBtagLoose = variables.nBtagLoose.fillna(0.0)
        variables.nBtagMedium = variables.nBtagMedium.fillna(0.0)

        variables.selection = (
            output.event_selection
            & (variables.nBtagLoose < 2)
            & (variables.nBtagMedium < 1)
        )

        # ------------------------------------------------------------#
        # Define categories
        # ------------------------------------------------------------#
        variables["c"] = ""
        variables.c[variables.selection & (variables.njets < 2)] = "ggh_01j"
        variables.c[
            variables.selection & (variables.njets >= 2) & (~vbf_cut)
        ] = "ggh_2j"
        variables.c[variables.selection & (variables.njets >= 2) & vbf_cut] = "vbf"

        # if 'dy' in dataset:
        #     two_jets_matched = np.zeros(numevents, dtype=bool)
        #     matched1 =\
        #         (jet1.matched_genjet.counts > 0)[two_jets[one_jet]]
        #     matched2 = (jet2.matched_genjet.counts > 0)
        #     two_jets_matched[two_jets] = matched1 & matched2
        #     variables.c[
        #         variables.selection &
        #         (variables.njets >= 2) &
        #         vbf_cut & (~two_jets_matched)] = 'vbf_01j'
        #     variables.c[
        #         variables.selection &
        #         (variables.njets >= 2) &
        #         vbf_cut & two_jets_matched] = 'vbf_2j'

        # --------------------------------------------------------------#
        # Fill outputs
        # --------------------------------------------------------------#

        variables.update({"wgt_nominal": weights.get_weight("nominal")})

        # All variables are affected by jet pT because of jet selections:
        # a jet may or may not be selected depending on pT variation.

        for key, val in variables.items():
            output.loc[:, pd.IndexSlice[key, variation]] = val

        return output

    def prepare_lookups(self):
        # Rochester correction
        rochester_data = txt_converters.convert_rochester_file(
            self.parameters["roccor_file"], loaduncs=True
        )
        self.roccor_lookup = rochester_lookup.rochester_lookup(rochester_data)

        # JEC, JER and uncertainties
        self.jec_factories, self.jec_factories_data = jec_factories(self.year)

        # Muon scale factors
        self.musf_lookup = musf_lookup(self.parameters)
        # Pile-up reweighting
        self.pu_lookups = pu_lookups(self.parameters)
        # Btag weights
        self.btag_lookup = BTagScaleFactor(
            self.parameters["btag_sf_csv"],
            BTagScaleFactor.RESHAPE,
            "iterativefit,iterativefit,iterativefit",
        )
        # STXS VBF cross-section uncertainty
        self.stxs_acc_lookups, self.powheg_xsec_lookup = stxs_lookups()

        # --- Evaluator
        self.extractor = extractor()

        # Z-pT reweigting (disabled)
        zpt_filename = self.parameters["zpt_weights_file"]
        self.extractor.add_weight_sets([f"* * {zpt_filename}"])
        if "2016" in self.year:
            self.zpt_path = "zpt_weights/2016_value"
        else:
            self.zpt_path = "zpt_weights/2017_value"
        # PU ID weights
        puid_filename = self.parameters["puid_sf_file"]
        self.extractor.add_weight_sets([f"* * {puid_filename}"])
        # Calibration of event-by-event mass resolution
        for mode in ["Data", "MC"]:
            label = f"res_calib_{mode}_{self.year}"
            path = self.parameters["res_calib_path"]
            file_path = f"{path}/{label}.root"
            self.extractor.add_weight_sets([f"{label} {label} {file_path}"])

        self.extractor.finalize()
        self.evaluator = self.extractor.make_evaluator()

        self.evaluator[self.zpt_path]._axes = self.evaluator[self.zpt_path]._axes[0]

        return

    def postprocess(self, accumulator):
        return accumulator
