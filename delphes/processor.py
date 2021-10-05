import pandas as pd
import awkward as ak
import coffea.processor as processor

from delphes.parameters import parameters
from delphes.muons import fill_muons
from delphes.jets import fill_jets


class DimuonProcessorDelphes(processor.ProcessorABC):
    def __init__(self, **kwargs):
        self.apply_to_output = kwargs.pop('apply_to_output', None)
        self._accumulator = processor.defaultdict_accumulator(int)

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    def process(self, df):
        output = pd.DataFrame({'event': df.Event.Number})
        output.index.name = 'entry'

        output['s'] = df.metadata['dataset']
        regions = df.metadata['regions']
        # channels = df.metadata['channels']
        output['lumi_wgt'] = df.metadata['lumi_wgt']
        output['mc_wgt'] = ak.to_pandas(df.Event.Weight)
        # There are multiple weights per event - need to figure this out
        # output['lhe_wgt'] = ak.to_pandas(df.Weight.Weight)
        output['year'] = 'snowmass'

        # Select muons
        muon_columns = ['PT', 'Eta', 'Phi', 'Charge', 'IsolationVar']
        muons = ak.to_pandas(df[parameters['muon_branch']][muon_columns])
        muons['selection'] = (
            (muons.PT > parameters["muon_pt_cut"]) &
            (abs(muons.Eta) < parameters["muon_eta_cut"]) &
            (muons.IsolationVar < parameters["muon_iso_cut"])
        )
        nmuons = muons[muons.selection].reset_index().groupby('entry')['subentry'].nunique()
        mm_charge = muons.loc[muons.selection, 'Charge'].groupby('entry').prod()

        muons = muons[muons.selection & (nmuons == 2)]
        mu1 = muons.loc[muons.PT.groupby('entry').idxmax()]
        mu2 = muons.loc[muons.PT.groupby('entry').idxmin()]
        mu1.index = mu1.index.droplevel('subentry')
        mu2.index = mu2.index.droplevel('subentry')
        pass_leading_pt = (mu1.PT > parameters["muon_leading_pt"])

        fill_muons(output, mu1, mu2)

        # Select electrons
        ele_columns = ['PT', 'Eta']
        electrons = ak.to_pandas(df[parameters['electron_branch']][ele_columns])
        electrons['selection'] = (
            (electrons.PT > parameters["electron_pt_cut"]) &
            (abs(electrons.Eta) < parameters["electron_eta_cut"])
        )
        electrons = electrons[electrons.selection]
        nelectrons = electrons.reset_index().groupby('entry')['subentry'].nunique().fillna(0)

        # Select jets
        jet_columns = ['PT', 'Eta', 'Phi', 'Mass']
        jets = ak.to_pandas(df[parameters['jet_branch']][jet_columns])
        jets['selection'] = (
            (jets.PT > parameters["jet_pt_cut"]) &
            (abs(jets.Eta) < parameters["jet_eta_cut"])
        )
        jets = jets[jets.selection]
        njets = jets.reset_index().groupby('entry')['subentry'].nunique()
        jets = jets.sort_values(['entry', 'PT'], ascending=[True, False])
        jets.index = pd.MultiIndex.from_arrays(
            [jets.index.get_level_values(0), jets.groupby(level=0).cumcount()],
            names=['entry', 'subentry']
        )
        jet1 = jets.loc[pd.IndexSlice[:, 0], :]
        jet2 = jets.loc[pd.IndexSlice[:, 1], :]
        jet1.index = jet1.index.droplevel('subentry')
        jet2.index = jet2.index.droplevel('subentry')

        fill_jets(output, jet1, jet2)

        # Event selection: two opposite-sign muons and no electrons
        output['nmuons'] = nmuons
        output['nelectrons'] = nelectrons
        output['njets'] = njets
        output[['nmuons', 'nelectrons', 'njets']] = output[['nmuons', 'nelectrons', 'njets']].fillna(0)

        output['event_selection'] = (
            (output.nmuons == 2) &
            (mm_charge == -1) &
            (output.nelectrons == 0) &
            pass_leading_pt
        )

        mass = output.dimuon_mass
        output['r'] = None
        output.loc[((mass > 76) & (mass < 106)), 'r'] = "z-peak"
        output.loc[((mass > 110) & (mass < 115.03)) |
                   ((mass > 135.03) & (mass < 150)), 'r'] = "h-sidebands"
        output.loc[((mass > 115.03) & (mass < 135.03)), 'r'] = "h-peak"

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
