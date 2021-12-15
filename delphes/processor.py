import pandas as pd
import awkward as ak
import coffea.processor as processor

from delphes.config.parameters import parameters
from delphes.muons import fill_muons
from delphes.jets import fill_jets


class DimuonProcessorDelphes(processor.ProcessorABC):
    def __init__(self, **kwargs):
        self.apply_to_output = kwargs.pop("apply_to_output", None)
        self._accumulator = processor.defaultdict_accumulator(int)

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    def process(self, df):
        # numevents = len(df)
        # dataset = df.metadata["dataset"]
        output = pd.DataFrame({"event": df.Event.Number})

        output.index.name = "entry"

        output["dataset"] = df.metadata["dataset"]
        regions = df.metadata["regions"]
        # channels = df.metadata['channels']
        output["lumi_wgt"] = float(df.metadata["lumi_wgt"])
        output["mc_wgt"] = ak.to_pandas(df.Event.Weight)
        # There are multiple weights per event - need to figure this out
        # output['lhe_wgt'] = ak.to_pandas(df.Weight.Weight)
        output["year"] = "snowmass"

        # Select muons
        muons = df[parameters["muon_branch"]]
        muon_filter = (
            (muons.pt > parameters["muon_pt_cut"])
            & (abs(muons.eta) < parameters["muon_eta_cut"])
            & (muons.IsolationVar < parameters["muon_iso_cut"])
        )
        nmuons = ak.to_pandas(ak.count(muons[muon_filter].pt, axis=1))

        mu_map = {"PT": "pt", "Eta": "eta", "Phi": "phi", "Charge": "charge"}
        muon_columns = ["PT", "Eta", "Phi", "Charge", "IsolationVar"]

        # Convert one column at a time to preserve event indices in Pandas
        muon_feature_list = []
        for col in muon_columns:
            muon_feature = df[parameters["muon_branch"]][col]
            val = ak.to_pandas(muon_feature[muon_filter])
            muon_feature_list.append(val)

        muons = pd.concat(muon_feature_list, axis=1)
        muons.columns = muon_columns
        muons.rename(columns=mu_map, inplace=True)

        mu1 = muons.loc[muons.pt.groupby("entry").idxmax()]
        mu2 = muons.loc[muons.pt.groupby("entry").idxmin()]
        mu1.index = mu1.index.droplevel("subentry")
        mu2.index = mu2.index.droplevel("subentry")
        pass_leading_pt = mu1.pt > parameters["muon_leading_pt"]
        fill_muons(output, mu1, mu2)

        output.mm_charge = output.mu1_charge * output.mu2_charge

        # Select electrons
        electrons = df[parameters["electron_branch"]]
        electrons = electrons[
            (electrons.pt > parameters["electron_pt_cut"])
            & (abs(electrons.eta) < parameters["electron_eta_cut"])
        ]
        nelectrons = ak.to_pandas(ak.count(electrons.pt, axis=1))

        # Select jets
        jets = df[parameters["jet_branch"]]
        mu_for_clean = df[parameters["muon_branch"]]
        mu_for_clean = mu_for_clean[
            (mu_for_clean.pt > parameters["muon_pt_cut"])
            & (mu_for_clean.IsolationVar < parameters["muon_iso_cut"])
        ]
        _, jet_mu_dr = jets.nearest(mu_for_clean, return_metric=True)
        jet_filter = (
            ak.fill_none(jet_mu_dr > parameters["min_dr_mu_jet"], True)
            & (jets.pt > parameters["jet_pt_cut"])
            & (abs(jets.eta) < parameters["jet_eta_cut"])
        )
        njets = ak.to_pandas(ak.count(jets[jet_filter].pt, axis=1))

        jet_map = {"PT": "pt", "Eta": "eta", "Phi": "phi", "Mass": "mass"}
        jet_columns = ["PT", "Eta", "Phi", "Mass"]

        jet_feature_list = []
        for col in jet_columns:
            jet_feature = df[parameters["jet_branch"]][col]
            val = ak.to_pandas(jet_feature[jet_filter])
            jet_feature_list.append(val)

        jets = pd.concat(jet_feature_list, axis=1)
        jets.columns = jet_columns
        jets.rename(columns=jet_map, inplace=True)

        jets = jets.sort_values(["entry", "pt"], ascending=[True, False])
        jets.index = pd.MultiIndex.from_arrays(
            [jets.index.get_level_values(0), jets.groupby(level=0).cumcount()],
            names=["entry", "subentry"],
        )
        jet1 = jets.loc[pd.IndexSlice[:, 0], :]
        jet2 = jets.loc[pd.IndexSlice[:, 1], :]
        jet1.index = jet1.index.droplevel("subentry")
        jet2.index = jet2.index.droplevel("subentry")

        fill_jets(output, jet1, jet2)

        # Event selection: two opposite-sign muons and no electrons
        output["nmuons"] = nmuons
        output["nelectrons"] = nelectrons
        output["njets"] = njets
        output[["nmuons", "nelectrons", "njets"]] = output[
            ["nmuons", "nelectrons", "njets"]
        ].fillna(0)

        output["event_selection"] = (
            (output.nmuons == 2)
            & (output.mm_charge == -1)
            & (output.nelectrons == 0)
            & pass_leading_pt
        )

        mass = output.dimuon_mass
        output["region"] = None
        output.loc[((mass > 76) & (mass < 106)), "region"] = "z-peak"
        output.loc[
            ((mass > 110) & (mass < 115.03)) | ((mass > 135.03) & (mass < 150)),
            "region",
        ] = "h-sidebands"
        output.loc[((mass > 115.03) & (mass < 135.03)), "region"] = "h-peak"

        output = output.loc[output.event_selection, :]
        output = output.reindex(sorted(output.columns), axis=1)

        output = output[output.region.isin(regions)]

        """
        input_evts = numevents
        output_evts = output.shape[0]
        out_yield = output.lumi_wgt.sum()
        out_vbf = output[
            (output.jj_mass>400) & (output.jj_dEta>2.5) & (output.jet1_pt>35) & (output.njets>=2)
        ].lumi_wgt.sum()
        out_ggh = out_yield - out_vbf

        print(f"\n{dataset}:    {input_evts}  ->  {output_evts};    yield = {out_ggh} (ggH) + {out_vbf} (VBF) = {out_yield}")
        """

        to_return = None
        if self.apply_to_output is None:
            to_return = output
        else:
            self.apply_to_output(output)
            to_return = self.accumulator.identity()

        return to_return

    def postprocess(self, accumulator):
        return accumulator
