import pandas as pd
import awkward as ak
import coffea.processor as processor

from delphes.parameters import parameters
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
        output = pd.DataFrame({"event": df.Event.Number})
        output.index.name = "entry"

        output["s"] = df.metadata["dataset"]
        regions = df.metadata["regions"]
        # channels = df.metadata['channels']
        output["lumi_wgt"] = float(df.metadata["lumi_wgt"])
        output["mc_wgt"] = ak.to_pandas(df.Event.Weight)
        # There are multiple weights per event - need to figure this out
        # output['lhe_wgt'] = ak.to_pandas(df.Weight.Weight)
        output["year"] = "snowmass"

        # Select muons
        muons = df[parameters["muon_branch"]]
        muons = muons[
            (muons.pt > parameters["muon_pt_cut"])
            & (abs(muons.eta) < parameters["muon_eta_cut"])
            & (muons.IsolationVar < parameters["muon_iso_cut"])
        ]
        nmuons = ak.count(muons.pt, axis=1)
        muons = muons[nmuons == 2]

        mu_map = {"PT": "pt", "Eta": "eta", "Phi": "phi", "Charge": "charge"}
        for old, new in mu_map.items():
            muons[new] = muons[old]
        muon_columns = ["pt", "eta", "phi", "charge", "IsolationVar"]
        muons = ak.to_pandas(muons[muon_columns])
        nmuons = ak.to_pandas(nmuons)

        mm_charge = muons.loc[:, "charge"].groupby("entry").prod()
        mu1 = muons.loc[muons.pt.groupby("entry").idxmax()]
        mu2 = muons.loc[muons.pt.groupby("entry").idxmin()]
        mu1.index = mu1.index.droplevel("subentry")
        mu2.index = mu2.index.droplevel("subentry")
        pass_leading_pt = mu1.pt > parameters["muon_leading_pt"]

        fill_muons(output, mu1, mu2)

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
        jets = jets[
            ak.fill_none(jet_mu_dr > parameters["min_dr_mu_jet"], True)
            & (jets.pt > parameters["jet_pt_cut"])
            & (abs(jets.eta) < parameters["jet_eta_cut"])
        ]
        njets = ak.to_pandas(ak.count(jets.pt, axis=1))

        jet_map = {"PT": "pt", "Eta": "eta", "Phi": "phi", "Mass": "mass"}
        for old, new in jet_map.items():
            jets[new] = jets[old]
        jet_columns = ["pt", "eta", "phi", "mass"]
        jets = ak.to_pandas(jets[jet_columns])

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
            & (mm_charge == -1)
            & (output.nelectrons == 0)
            & pass_leading_pt
        )

        mass = output.dimuon_mass
        output["r"] = None
        output.loc[((mass > 76) & (mass < 106)), "r"] = "z-peak"
        output.loc[
            ((mass > 110) & (mass < 115.03)) | ((mass > 135.03) & (mass < 150)), "r"
        ] = "h-sidebands"
        output.loc[((mass > 115.03) & (mass < 135.03)), "r"] = "h-peak"

        output = output.loc[output.event_selection, :]
        output = output.reindex(sorted(output.columns), axis=1)

        output = output[output.r.isin(regions)]

        # print(output.isna().sum()[output.isna().sum()>0])

        to_return = None
        if self.apply_to_output is None:
            to_return = output
        else:
            self.apply_to_output(output)
            to_return = self.accumulator.identity()

        return to_return

    def postprocess(self, accumulator):
        return accumulator
