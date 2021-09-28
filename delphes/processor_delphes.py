import sys
import awkward
import awkward as ak
import pandas as pd

# https://github.com/kratsg/coffea/tree/feat/nanodelphes
sys.path.insert(0, "/home/dkondra/coffea_delphes/coffea/")

import coffea.processor as processor

# from python.weights import Weights

from delphes.parameters import parameters
from delphes.variables import variables
from delphes.muons import fill_muons


class DimuonProcessorDelphes(processor.ProcessorABC):
    def __init__(self, **kwargs):
        self.apply_to_output = kwargs.pop('apply_to_output', None)

        self._accumulator = processor.defaultdict_accumulator(int)

        self.parameters = parameters

        # --- Define regions and channels used in the analysis ---#
        self.regions = ['z-peak', 'h-sidebands', 'h-peak']
        # self.channels = ['ggh_01j', 'ggh_2j', 'vbf']
        self.channels = ['vbf', 'vbf_01j', 'vbf_2j']

        # self.lumi_weights = self.samp_info.lumi_weights

        self.vars_to_save = set([v.name for v in variables])

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    def process(self, df):
        dataset = df.metadata['dataset']
        # numevents = len(df)

        output = pd.DataFrame({'event': df.Event.Number})
        # print(df.Event.CrossSection)
        output.index.name = 'entry'

        muon_columns = ['PT', 'Eta', 'Phi', 'Charge']
        # missing: ptErr, mass, muonID, isolation
        # (there are 'IsolationVar','IsolationVarRhoCorr' though)

        muons = ak.to_pandas(df.Muon[muon_columns])

        muons['selection'] = (
            (muons.PT > self.parameters["muon_pt_cut"]) &
            (abs(muons.Eta) <
             self.parameters["muon_eta_cut"])
        )

        nmuons = muons[muons.selection].reset_index()\
            .groupby('entry')['subentry'].nunique()

        mm_charge = muons.loc[muons.selection, 'Charge']\
            .groupby('entry').prod()

        output['two_muons'] = (nmuons == 2)
        output['event_selection'] = (
            (nmuons == 2) &
            (mm_charge == -1)
        )

        muons = muons[muons.selection & (nmuons == 2)]
        mu1 = muons.loc[muons.PT.groupby('entry').idxmax()]
        mu2 = muons.loc[muons.PT.groupby('entry').idxmin()]
        mu1.index = mu1.index.droplevel('subentry')
        mu2.index = mu2.index.droplevel('subentry')

        pass_leading_pt = (
            mu1.PT > self.parameters["muon_leading_pt"]
        )

        output['pass_leading_pt'] = pass_leading_pt
        output['event_selection'] = (
            output.event_selection & output.pass_leading_pt
        )

        fill_muons(self, output, mu1, mu2)

        mass = output.dimuon_mass
        output['r'] = None
        output.loc[((mass > 76) & (mass < 106)), 'r'] = "z-peak"
        output.loc[((mass > 110) & (mass < 115.03)) |
                   ((mass > 135.03) & (mass < 150)),
                   'r'] = "h-sidebands"
        output.loc[((mass > 115.03) & (mass < 135.03)), 'r'] = "h-peak"
        output['s'] = dataset
        output['year'] = 'snowmass'

        output = output.loc[output.event_selection, :]
        output = output.reindex(sorted(output.columns), axis=1)

        output = output[output.r.isin(self.regions)]

        to_return = None
        if self.apply_to_output is None:
            to_return = output
        else:
            self.apply_to_output(output)
            to_return = self.accumulator.identity()

        return to_return

    def postprocess(self, accumulator):
        return accumulator
